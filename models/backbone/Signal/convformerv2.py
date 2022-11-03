import torch
import torch.nn.functional as F
import torch.nn as nn
from models.backbone.Signal.localvit import h_swish
from models.backbone.Signal.maxvit import Depthwise_conv_3x3

"""
the pytorch version of convformer-nse
Original Author: SongYu Han
Reported by 'Convformer-NSE: A Novel End-to-End Gearbox Fault Diagnosis Framework Under Heavy Noise Using Joint Global and Local Information'
doi: 10.1109/TMECH.2022.3199985.

"""


class donsample_conv(nn.Module):
    def __init__(self, k, s, c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=c, out_channels=c, kernel_size=k, stride=s, padding=int(k / 2))
        self.act1 = h_swish()
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(in_channels=c, out_channels=2 * c, kernel_size=3, stride=1, padding=1)
        self.act2 = h_swish()
        self.bn2 = nn.BatchNorm1d(2 * c)
        self.conv3 = nn.Conv1d(in_channels=2 * c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.act3 = h_swish()
        self.bn3 = nn.BatchNorm1d(c)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x)
        # [b, c, m] ==> [b, m, c]
        x = x.transpose(-2, -1)
        return x


class conv_projection(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=in_channels * ratio,
                               stride=1,
                               kernel_size=3,
                               padding=int(3 / 2))
        self.act1 = h_swish()
        self.conv2 = nn.Conv1d(in_channels=in_channels * ratio,
                               out_channels=in_channels,
                               stride=1,
                               kernel_size=1,
                               padding=0)
        self.act2 = h_swish()
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        b, c, n = x.shape[0], x.shape[1], x.shape[2]
        short_cut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = short_cut + x
        x = x.reshape(b, -1, c)
        x = self.norm(x)
        return x.view(b, c, -1)


class Sparse_Attention(nn.Module):
    def __init__(self,
                 dim,
                 conv_k, conv_s,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 proj_drop_ratio=0.,
                 attn_drop_ratio=0):
        super().__init__()
        k, s = conv_k, conv_s
        self.num_heads = num_heads
        self.dim = dim
        self.heads_dim = dim // num_heads
        self.scale = qk_scale or self.heads_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.k_conv_using = nn.Sequential(
            nn.Conv1d(in_channels=self.dim,
                      out_channels=self.dim,
                      kernel_size=k,
                      stride=s,
                      padding=int(k / 2)),
            nn.BatchNorm1d(self.dim),
            h_swish())
        self.v_conv_using = nn.Sequential(
            nn.Conv1d(in_channels=self.dim,
                      out_channels=self.dim,
                      kernel_size=k,
                      stride=s,
                      padding=int(k / 2)),
            nn.BatchNorm1d(self.dim),
            h_swish())
        self.liner_proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop_ratio, inplace=True)
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, x):
        b, n, c = x.shape
        # [b, n, c] ==> [b, n, 3*dim]
        short_cut = x
        qkv = self.qkv(x)
        # [b, n, 3*dim] ==> [b, n, 3, num_heads, heads_dim] ==> [3, b, num_heads,n,heads_dim]
        qkv = qkv.reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        # [b, num_heads, n, heads_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [b, num_heads, n, heads_dim] => [b*num_heads, n, heads_dim] ==>[b*num_heads, heads_dim, n]
        k = k.reshape(-1, n, c).transpose(-2, -1)
        v = v.reshape(-1, n, c).transpose(-2, -1)
        # [b*num_heads, heads_dim, n] ==>[b*num_heads,heads_dim, m]
        k = self.k_conv_using(k)
        v = self.v_conv_using(v)
        # [b * num_heads, heads_dim, m] ==>[b, num_heads, m, heads_dim]
        k = k.reshape(b, self.num_heads, self.heads_dim, -1)
        v = v.reshape(b, self.num_heads, self.heads_dim, -1).permute(0, 1, 3, 2)
        # [b, num_heads, n, heads_dim] @ p[b, num_heads, heads_dim, m] ==> [b, num_heads, n, m]
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # [b, num_heads, n, m] @[b, num_heads, m, heads_dim] ==> [b, num_heads, n, heads_dim] ==>[b, n, num_heads, dim] ==> [b, n, dim]
        x = (attn @ v).transpose(1, 2).reshape(b, n, self.dim)
        x = self.liner_proj(x)
        x = short_cut + x
        x = self.proj_drop(x)
        x = self.norm(x)
        return x.view(b, c, n)


class Convformer_Block(nn.Module):
    def __init__(self,
                 k, s, c,
                 conv_k, conv_s,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 proj_drop_ratio=0.,
                 attn_drop_ratio=0):
        super().__init__()
        self.conv_module = donsample_conv(k=k, s=s, c=c)
        self.attention = Sparse_Attention(dim=c,
                                          conv_k=conv_k,
                                          conv_s=conv_s,
                                          num_heads=num_heads,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,
                                          proj_drop_ratio=proj_drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio
                                          )
        self.conv_proj = conv_projection(in_channels=c)

    def forward(self, x):
        x = self.conv_module(x)
        x = self.attention(x)
        x = self.conv_proj(x)
        return x


class ConvformerStage(nn.Module):
    def __init__(self,
                 depth,
                 k, s, c,
                 conv_k, conv_s,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 proj_drop_ratio=0.,
                 attn_drop_ratio=0):
        super().__init__()
        self.blokcs = nn.Sequential(*[
            Convformer_Block(k=k, s=s if i == 0 else 1, c=c,
                             conv_k=conv_k,
                             conv_s=conv_s,
                             num_heads=num_heads,
                             qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             proj_drop_ratio=proj_drop_ratio,
                             attn_drop_ratio=attn_drop_ratio)
            for i in range(depth)
        ])

    def forward(self, x):
        return self.blokcs(x)


class Avg_max_channel_attention(nn.Module):
    def __init__(self, in_channel, ratio):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(nn.Linear(in_features=in_channel,
                                           out_features=in_channel * ratio),
                                 nn.ReLU(True),
                                 nn.Linear(in_features=in_channel * ratio,
                                           out_features=in_channel),
                                 nn.Softmax(dim=-1))

    def forward(self, x):
        # [b, c, n] ==> [b, c, 1]
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        # [b, c, 1] ==> [b, 1, c]
        x_avg = x_avg.transpose(-2, -1)
        x_max = x_max.transpose(-2, -1)
        x_avg_attn = self.mlp(x_avg)
        x_max_attn = self.mlp(x_max)
        x_avg_attn = x_avg_attn.transpose(-2, -1)
        x_max_attn = x_max_attn.transpose(-2, -1)
        return x_avg_attn * x + x_max_attn * x


class Convformer_nse(nn.Module):
    def __init__(self, cfg,
                 in_c,
                 num_cls=8,
                 h_args=None,
                 use_init=True,
                 use_last=True,
                 downscale=True):
        '''

        :param cfg: the parameters of each scale convformer-nse
        :param in_c: the input channels of data
        :param h_args: the hidden layers num and neurons of classifier
        :param use_init: use weight initializing or not
        :param use_last: use the lats features or not
        :param num_cls: number of classes for classification head
        '''
        super().__init__()
        self.use_last = use_last
        self.stem = Depthwise_conv_3x3(in_c=in_c,
                                       out_channels=cfg[0][3],
                                       downscale=downscale)
        self.stage1, in_c0 = self._make_layers(cfg[0])
        self.stage2, in_c1 = self._make_layers(cfg[1])
        self.stage3, in_c2 = self._make_layers(cfg[2])
        self.patch_merging1 = nn.Sequential(
            nn.Conv1d(in_channels=cfg[0][3],
                      out_channels=cfg[1][3],
                      kernel_size=1,
                      stride=1,
                      padding=0),
            h_swish(),
            nn.BatchNorm1d(cfg[1][3]))
        self.stage2, in_c1 = self._make_layers(cfg[1])
        self.patch_merging2 = nn.Sequential(
            nn.Conv1d(in_channels=cfg[1][3],
                      out_channels=cfg[2][3],
                      kernel_size=1,
                      stride=1,
                      padding=0),
            h_swish(),
            nn.BatchNorm1d(cfg[2][3]))

        self.stage3, in_c2 = self._make_layers(cfg[2])
        if self.use_last:
            self.nse = Avg_max_channel_attention(in_channel=in_c2, ratio=4)
            self.last_channels = in_c2
        else:
            self.nse = Avg_max_channel_attention(in_channel=in_c0 + in_c1 + in_c2, ratio=4)
            self.last_channels = in_c0 + in_c1 + in_c2
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.ModuleList()
        if not h_args:
            self.classifier.append(nn.Linear(self.last_channels, num_cls))
            self.classifier.append(nn.Softmax(dim=-1))
        else:
            for i in range(len(h_args)):
                if i == 0:
                    self.classifier.append(nn.Linear(self.last_channels, h_args[i]))
                    self.classifier.append(h_swish())
                    # self.classifier.append(nn.Dropout(p=0))
                else:
                    self.classifier.append(nn.Linear(h_args[i - 1], h_args[i]))
                    self.classifier.append(h_swish())
                    # self.classifier.append(nn.Dropout(p=0))
            self.classifier.append(nn.Linear(h_args[-1], num_cls))
            self.classifier.append(nn.Softmax(dim=-1))
        if use_init:
            self._initialize_weights()

    def forward(self, x):
        b = x.shape[0]
        x = self.stem(x)
        x = self.stage1(x)
        f0 = x
        x = self.patch_merging1(x)
        x = self.stage2(x)
        f1 = x
        x = self.patch_merging2(x)
        x = self.stage3(x)
        f2 = x
        if self.use_last:
            x = self.nse(x)
        else:
            f0 = F.interpolate(f0, size=f2.size()[-1])
            f1 = F.interpolate(f1, size=f2.size()[-1])
            x = torch.cat([f0, f1, f2], dim=1)
            x = self.nse(x)
        # [b, c1, n] ==> [b, c1, 1]
        x = self.avg_pool(x)
        x = x.view(b, -1)
        for module in self.classifier:
            x = module(x)
        return x

    @staticmethod
    def _make_layers(params, use_bn=True):
        layers = []
        layers += [ConvformerStage(depth=params[0], k=params[1], s=params[2],
                                    c=params[3], conv_s=params[4], conv_k=params[5], num_heads=params[-1])]

        return nn.Sequential(*layers), params[3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# depth, kernel_size, strides (downsampling), dim, kernel_size, strides(Attention), num_heads,
cfgs = {"S": [[1, 2, 5, 16, 2, 2, 2], [1, 2, 3, 32, 2, 2, 4], [1, 2, 3, 64, 2, 2, 4]],
        "M": [],
        "B": []}

"""
def make_layers(params, use_bn=True):
    layers = []
    layers += [Convformer_Stage(depth=params[0], k=params[1], s=params[2],
                                c=params[3], conv_s=params[4], conv_k=params[5], num_heads=params[-1])]

    return nn.Sequential(*layers), params[3]


layer, c0 = make_layers(params=cfgs["S"][1])
tensor = torch.rand(2, 32, 1024)
print(layer(tensor).shape)
"""


def convoformer_v2_small(h_args, in_c, num_cls):
    model = Convformer_nse(cfg=cfgs["S"],
                           h_args=h_args,
                           in_c=in_c,
                           num_cls=num_cls)
    return model


def convoformer_v2_middle(h_args, in_c, num_cls):
    model = Convformer_nse(cfg=cfgs["M"],
                           h_args=h_args,
                           in_c=in_c,
                           num_cls=num_cls)
    return model


def convormer_v2_big(h_args, in_c, num_cls):
    model = Convformer_nse(cfg=cfgs["B"],
                           h_args=h_args,
                           in_c=in_c,
                           num_cls=num_cls)
    return model


if __name__ == "__main__":
    model = convoformer_v2_small(h_args=None, in_c=2, num_cls=8)
    tensor = torch.rand(5, 2, 1024)
    print(model(tensor).shape)
