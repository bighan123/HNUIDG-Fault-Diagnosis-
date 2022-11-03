import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.Signal.vit import DropPath, Mlp, Attention, PatchEmbed


class PatchEmbed_for_uniformer(PatchEmbed):
    def __init__(self, data_size, in_c, patch_size, norm_layer, out_dim):
        super().__init__(data_size, in_c, patch_size, norm_layer)
        self.projection = nn.Conv1d(in_c, out_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(out_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        # [B, C, N] ==> [B, embed_dim, grid_size]
        x = self.projection(x)
        # [B,embed_dim, grid_size] ==> [B, grid_size, embed_dim]
        x = x.transpose(1,2)
        x = self.norm(x)
        return x.transpose(1,2)

class CMlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# The local MHRA
class local_MHRA(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()
        self.dpe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm1d(dim)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.attention = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm1d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.dpe(x)
        x = x + self.drop_path(self.conv2(self.attention(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class global_MHRA(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()
        self.dpe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attention = Attention(dim=dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   proj_drop_ratio=drop,
                                   attn_drop_ratio=attn_drop,
                                   )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.dpe(x)
        B, C, N = x.shape
        x = x.transpose(-1, -2)
        if self.layer_scale:
            x = x + self.drop_path(self.gamma1 * self.attention(self.norm1(x)))
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attention(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(-1, -2)
        return x


class UniformerStage(nn.Module):
    def __init__(self,
                 dim,
                 depths,
                 num_heads,
                 MHRA=local_MHRA,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None
                 ):
        super().__init__()
        self.blocks = nn.Sequential(*[
            MHRA(dim=dim,
                 num_heads=num_heads,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 drop=drop,
                 attn_drop=attn_drop,
                 drop_path=drop_path,
                 act_layer=act_layer,
                 norm_layer=norm_layer,
                 layer_scale=layer_scale
                 ) for _ in range(depths)
        ])

    def forward(self, x):
        return self.blocks(x)


class Uniformer(nn.Module):
    def __init__(self, data_size, in_c, h_args, num_cls,
                 dims,
                 depths,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None
                 ):
        super().__init__()
        self.blocks = []
        for idx, (dim, depth, num_head) in enumerate(zip(dims, depths, num_heads)):
            blocks = UniformerStage(dim=dim, depths=depth, num_heads=num_head,
                                    MHRA=global_MHRA if idx > 1 else local_MHRA,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop,
                                    attn_drop=attn_drop,
                                    drop_path=drop_path,
                                    act_layer=act_layer,
                                    norm_layer=norm_layer,
                                    layer_scale=layer_scale)

            patch_embed = PatchEmbed_for_uniformer(data_size=data_size // 2 ** idx if idx == 0 else 2 ** (idx + 1),
                                                   patch_size=4 if idx == 0 else 2,
                                                   in_c=in_c if idx == 0 else dims[idx - 1], out_dim=dim,
                                                   norm_layer=nn.LayerNorm)
            self.blocks += [patch_embed, blocks]

        self.classifier = nn.ModuleList()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_channels = dims[-1]
        if not h_args:
            self.classifier.append(nn.Linear(self.last_channels, num_cls))
            self.classifier.append(nn.Softmax(dim=-1))
        else:
            for i in range(len(h_args)):
                if i == 0:
                    self.classifier.append(nn.Linear(self.last_channels, h_args[i]))
                else:
                    self.classifier.append(nn.Linear(h_args[i - 1], h_args[i]))
            self.classifier.append(nn.Linear(h_args[-1], num_cls))
            self.classifier.append(nn.Softmax(dim=-1))

        self.blocks = nn.Sequential(*self.blocks)
        self.norm = nn.BatchNorm1d(dims[-1])
        self._init_weights()

    def forward(self, x):
        b = x.shape[0]
        x = self.blocks(x)
        x = self.norm(x)
        x = self.avg_pool(x).view(b, -1)
        for module in self.classifier:
            x = module(x)
        return x

    def get_classifier(self):
        return self.classifier

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

def uniformer_tiny(data_size, in_c, h_args, num_cls):
    model = Uniformer(data_size=data_size,
                      in_c=in_c,
                      h_args=h_args,
                      num_cls=num_cls,
                      depths=[3,4,8,3],
                      dims=[64,128,256,512],
                      num_heads=[8,8,8,8],
                      mlp_ratio=4,
                      qkv_bias=True)
    return model




