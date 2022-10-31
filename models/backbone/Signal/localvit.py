import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.Signal.vit import DropPath, Attention, VisionTransformer

'''
The pytorch version (deal with one dim data) of local vit
https://arxiv.org/abs/2104.05707
Original code: https://github.com/ofsoundof/LocalViT/blob/main/models/localvit.py
'''


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, -1)
        y = self.fc(y).view(b, c, 1)
        return x * y


class LocalityFeedForward(nn.Module):
    '''
    Args:
        expand_ratio: (int, optional), the expansion ratio of the hidden dimension
        act: (str, optional), the activation function
            hs: h_swish and SE module
            relu:ReLU
        channel_module: the channel attention module used
            se: SE layer module
            eca： ECA layer module
            "": do not use channel attention module
        reduction: reduction rate in SE module
        use_dw_conv: use depth-wise convolution or not
        dw_conv_first: place depth-wise convolution as the first layer

    '''

    def __init__(self, in_c, out_channels,
                 stride, expand_ratio=4,
                 act="hs",
                 channel_module="se",
                 reduction=4,
                 wo_dw_conv=False,
                 dw_conv_first=False):
        super().__init__()
        hidden_dim = int(in_c * expand_ratio)
        layers = []
        layers.extend([nn.Conv1d(in_c,
                                 hidden_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       h_swish() if act == "hs" else nn.ReLU6(True)])
        if not wo_dw_conv:
            dp_layer = [nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                  padding=int(3 // 2), groups=hidden_dim, bias=False),
                        nn.BatchNorm1d(hidden_dim),
                        h_swish() if act == "hs" else nn.ReLU6(True)]
            if dw_conv_first:
                layers = dp_layer + layers
            else:
                layers.extend(dp_layer)

        self.downsample = nn.Conv1d(in_channels=in_c,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=stride,
                                    padding=0,
                                    bias=False) if not wo_dw_conv or in_c != out_channels else nn.Identity
        if channel_module != " ":
            if channel_module == "se":
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif channel_module == "eca":
                layers.append(ECALayer(hidden_dim, sigmoid=True))
            else:
                raise NotImplementedError('channel attention type {} is not implemented'.format(channel_module))
        layers.extend([nn.Conv1d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                       nn.BatchNorm1d(out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.downsample(x) + self.conv(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio,
                 qkv_bias=False, qkv_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.,
                 drop_path=0., act="hs", channel_module="se",
                 reduction=4, wo_dw_conv=False, dw_conv_first=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(dim=dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qkv_scale,
                                   attn_drop_ratio=attn_drop_ratio,
                                   proj_drop_ratio=proj_drop_ratio)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()
        self.replaced_mlp = LocalityFeedForward(in_c=dim, out_channels=dim,
                                                stride=1, expand_ratio=mlp_ratio,
                                                act=act, channel_module=channel_module,
                                                reduction=reduction, wo_dw_conv=wo_dw_conv,
                                                dw_conv_first=dw_conv_first)

    def forward(self, x):
        B, N, C = x.shape
        d = self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.attention(self.norm1(x)))
        # [B, 1, dim], [B, N - 1, dim]
        cls_token, x = torch.split(x, [1, N - 1], dim=1)
        # [B, dim, N - 1]
        x = x.transpose(1, 2)
        # [B, N - 1, dim]
        x = self.replaced_mlp(x).transpose(1, 2)
        x = torch.cat([cls_token, x], dim=1)
        return x


class LocalVisionTransformer(VisionTransformer):
    def __init__(self, data_size, in_c, num_cls,
                 cfgs, h_args=None, mlp_ratio=4.0,
                 qkv_bias=False, qkv_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 norm_layer=None, act_layer=None, use_init=True,
                 act="hs", channel_module="se", reduction=4, wo_dw_conv=False,
                 dw_conv_first=False):
        super().__init__(data_size, in_c, num_cls, cfgs, h_args, mlp_ratio, qkv_bias,
                         qkv_scale, drop_ratio, attn_drop_ratio, drop_path_ratio,
                         norm_layer, act_layer, use_init)
        depth, num_heads, patch_size = cfgs[0], cfgs[1], cfgs[2]
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.num_features = self.embed_dim = patch_size * in_c
        self.blocks = nn.Sequential(*[
            Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qkv_scale=qkv_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,
                  drop_path=dpr[i], act=act, channel_module=channel_module, reduction=reduction, wo_dw_conv=wo_dw_conv,
                  dw_conv_first=dw_conv_first)
            for i in range(depth)])


cfgs = {'base': [12, 8, 16],
        "middle_1": [12, 16, 16],
        'middle_2': [12, 16, 32],
        'large_1': [24, 16, 16],
        'large_2': [24, 16, 32]}

'''
type1: activation hs + channel attention se
type2: activation hs + channel attention eca
type3: activation ReLU + channel attention se
type4: activation ReLU + channel attention eca
'''
def localvit_base_patch16_type1(data_size, in_c, num_cls, h_args, cfgs=cfgs["base"]):
    model = LocalVisionTransformer(data_size =data_size,
                                   in_c=in_c,
                                   num_cls=num_cls,
                                   h_args=h_args,
                                   cfgs=cfgs)
    return model
def localvit_base_patch16_type2(data_size, in_c, num_cls, h_args, cfgs=cfgs["base"]):
    model = LocalVisionTransformer(data_size = data_size,
                                   in_c=in_c,
                                   num_cls=num_cls,
                                   h_args=h_args,
                                   cfgs=cfgs,
                                   channel_module="eca")
    return model

def localvit_middle1_patch16_type1(data_size, in_c, num_cls, h_args, cfgs=cfgs["middle_1"]):
    model = LocalVisionTransformer(data_size = data_size,
                                   in_c=in_c,
                                   num_cls=num_cls,
                                   h_args=h_args,
                                   cfgs=cfgs)
    return model

def localvit_middle2_patch32_type1(data_size, in_c, num_cls, h_args, cfgs=cfgs["middle_2"]):
    model = LocalVisionTransformer(data_size = data_size,
                                   in_c=in_c,
                                   num_cls=num_cls,
                                   h_args=h_args,
                                   cfgs=cfgs)
    return model

if __name__ == "__main__":
    model = localvit_middle2_patch32_type1(data_size=1024, in_c=3, num_cls=9,h_args=None)
    tensor = torch.rand(2, 3, 1024)
    print(model(tensor).shape)