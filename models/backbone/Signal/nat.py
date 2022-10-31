import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.Signal.vit import Mlp, DropPath

'''
The pytorch version of NAT(process one dimension data
This work is inspired by NAT,NNeighborhood Attention Attention without CUDA kernel
Original code link:https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/classification/nat.py
'''


class NeighborhoodAttention(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 num_heads=8,
                 qkv_bias=True,
                 qk_sacle=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dilation=1):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_sacle or self.head_dim ** -0.5
        self.window_size = self.kernel_size * dilation
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.pos_encoding = nn.Parameter(torch.zeros(self.window_size, self.window_size))
        torch.nn.init.trunc_normal_(self.pos_encoding, mean=0, std=0.01, a=-2, b=2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        pad_l = pad_r = self.window_size // 2
        # [B, N, C] ==> [B, N_, C]
        x = F.pad(x, (0, 0, pad_l, pad_r))
        _, N_, _ = x.shape
        # [B, N_, C] ==> [B, N_, 3C] ==> [B, N_, 3, num_heads, head_dim] ==> [3, B, num_heads, N_, head_dim]
        qkv = self.qkv(x).reshape(B, N_, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # [3, B, num_heads, N, head_dim, window_size] ==> [3, B, num_heads, N, window_size, head_dim]
        qkv = qkv.unfold(size=self.window_size, dimension=3, step=1).transpose(-1, -2)
        # [B, num_heads, N, window_size, head_dim]
        q, k, v = qkv.unbind(0)
        # [B, num_heads, N, window_size, window_size]
        q = q * self.scale
        attn = (q @ k.transpose(-1, -2)) + self.pos_encoding
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # [B, num_heads, N, window_size, head_dim]
        # [B, num_heads, N, head_dim]
        value = (attn @ v).transpose(-1, -2).sum(-1)
        value = value.transpose(-1, -2).reshape(B, N, C)
        value = self.proj(value)
        value = self.proj_drop(value)
        return value


class ConvTokenizer(nn.Module):
    def __init__(self,
                 in_c,
                 dim=64,
                 norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_c, dim // 2,
                      kernel_size=3, stride=2, padding=1),
            nn.Conv1d(dim // 2, dim,
                      kernel_size=3, stride=2, padding=1))
        self.norm = norm_layer(dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        # [B, C, N] ==> [B, N, C]
        x = self.proj(x).permute(0, 2, 1)
        return self.norm(x)


class ConvDownsampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Conv1d(dim, dim * 2,
                                   kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x)
        return x


class NATLayer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 kernel_sizes,
                 dilation=1,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attention = NeighborhoodAttention(dim=dim,
                                               kernel_size=kernel_sizes,
                                               dilation=dilation,
                                               num_heads=num_heads,
                                               qkv_bias=qkv_bias,
                                               qk_sacle=qk_scale,
                                               attn_drop=attn_drop,
                                               proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NATStage(nn.Module):
    def __init__(self,
                 dim,
                 depths,
                 num_heads,
                 kernel_sizes,
                 dilations=1,
                 downsample=True,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depths = depths

        self.stage = nn.Sequential(*[
            NATLayer(dim=dim,
                     num_heads=num_heads,
                     kernel_sizes=kernel_sizes,
                     dilation=None if dilations is None else 1,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     drop=drop,
                     attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer) for i in range(depths)

        ])
        self.downsample = nn.Identity() if not downsample else ConvDownsampler(dim=dim)

    def forward(self, x):
        for stage in self.stage:
            x = stage(x)
        x = self.downsample(x)
        return x


class NAT(nn.Module):
    def __init__(self,
                 in_c,
                 h_args,
                 num_cls,
                 dim,
                 depths,
                 num_heads,
                 kernel_sizes,
                 mlp_ratio=4,
                 drop_path_ratio=0.2,
                 dilations=1,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 norm_layer=nn.LayerNorm,
                 use_init=True):
        super().__init__()
        self.num_cls = num_cls
        self.num_levels = len(depths)
        self.dim = dim
        self.num_features = self.last_channels = int(dim * 2 ** (self.num_levels - 1))
        self.mlp_ratio = mlp_ratio

        self.path_embedding = ConvTokenizer(in_c=in_c, dim=dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, sum(depths))]

        self.stage = []
        for idx, (depth, num_head, kernel_size) in enumerate(zip(depths, num_heads, kernel_sizes)):
            stage = NATStage(dim=int(dim * 2 ** idx),
                             depths=depth,
                             num_heads=num_head,
                             kernel_sizes=kernel_size,
                             dilations=1,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             drop=drop_ratio,
                             attn_drop=attn_drop_ratio,
                             drop_path=dpr[sum(depths[:idx]):sum(depths[:idx + 1])],
                             norm_layer=norm_layer,
                             downsample=(idx < self.num_levels - 1))
            self.stage.append(stage)
        self.norm = norm_layer(self.num_features)
        self.stage = nn.ModuleList(self.stage)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.ModuleList()
        if h_args is None:
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
        if use_init:
            self._initialize_weights()
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

    def forward(self, x):
        b = x.shape[0]
        x = self.path_embedding(x)
        x = self.pos_drop(x)
        for stage in self.stage:
            x = stage(x)
        # [B, N, C] ==> [B, C, N]
        x = self.norm(x).transpose(1,2)
        # [B, C, 1]
        x = self.avgpool(x)
        x = x.view(b, -1)
        for module in self.classifier:
            x = module(x)
        return x

def nat_tiny(in_c, h_args, num_cls):
    model = NAT(in_c=in_c,
                h_args=h_args,
                num_cls=num_cls,
                dim=64,
                depths=[2,3,5,3],
                num_heads=[2,4,8,8],
                kernel_sizes=[3,5,7,7])
    return model

def nat_small(in_c, h_args, num_cls):
    model = NAT(in_c=in_c,
                h_args=h_args,
                num_cls=num_cls,
                dim=64,
                depths=[2,4,7,3],
                num_heads=[2,4,8,16],
                kernel_sizes=[3,5,7,7])
    return model

def nat_base(in_c, h_args, num_cls):
    model = NAT(in_c=in_c,
                h_args=h_args,
                num_cls=num_cls,
                dim=64,
                depths=[3,5,15,8],
                num_heads=[4,8,16,16],
                kernel_sizes=[5,7,11,11])
    return model
