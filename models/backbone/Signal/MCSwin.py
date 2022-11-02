import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.Signal.vit import Mlp, DropPath
from models.backbone.Signal.maxvit import window_partition, window_reverse
from models.backbone.Signal.nat import ConvDownsampler

'''
The pytorch version of MCSwin
'Multi-channel Calibrated Transformer with Shifted Windows for few-shot fault diagnosis under sharp speed variation'
ISA Transactions
We use learning absolute position embedding to replace the partially enhanced encoding, cause we did not find the 
detailed information about this in the paper.
'''


class Convolutional_Embedding(nn.Module):
    def __init__(self,
                 in_c,
                 kernel_sizes,
                 strides,
                 out_channels):
        super().__init__()
        self.in_c = in_c
        self.norm = nn.BatchNorm1d(in_c)
        self.layers = []
        for idx, (kernel_size, stride, out_channel) in enumerate(zip(kernel_sizes, strides, out_channels)):
            if idx == 0:
                layer = nn.Conv1d(in_channels=in_c, out_channels=out_channels[idx],
                                  kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2))
            else:
                layer = nn.Conv1d(in_channels=out_channels[idx - 1], out_channels=out_channel,
                                  kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2))
            self.layers += [layer, nn.BatchNorm1d(out_channel), nn.ReLU(True)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.norm(x)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == 0:
                f = x
        f = f.transpose(-1, -2)
        f = F.interpolate(f, size=(x.shape[1]))
        x = x + f.transpose(-1, -2)
        return x


class PatchEmerging(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_size,
                 stride):
        super().__init__()
        self.path_embedding = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=patch_size, stride=stride)

    def forward(self, x):
        return self.path_embedding(x)


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with learning absolute position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_drop=0.,
                 attn_drop=0., ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_heads, window_size, window_size))
        torch.nn.init.trunc_normal_(self.pos_embedding, mean=0, std=0.001, a=-2, b=2)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # B_: batch_size * num_windows
        # N： window_size(计算一个窗口内的注意力)
        # [B_, N, C] ==> [B_, N, 3C] => [B_, N, 3, self.num_heads, heads_dim] ==> [3, B_, self.num_heads, N, heads_dim]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) + self.pos_embedding

        if mask is not None:
            # mask shape [nw, window_size, window_size]
            nw = mask.shape[0]  # num_windows
            # [B_, num_heads, window_size, window_size] ==> [batch_size, num_windows, num_heads, N, N]
            attn = attn.view(B_ // nw, nw, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # [B_, num_heads, window_size, window_size]
        attn = self.attn_drop(attn)
        # [B_, num_heads, window_size, window_size] ==> [B_, num_heads, window_size, head_dim]
        # ==>[B_, window_size, num_heads, head_dim] ==> [B_,]
        value = (attn @ v).transpose(1, 2).reshape(-1, N, C)
        value = self.proj(value)
        value = self.proj_drop(value)

        return value


class MCSwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=8,
                 shift_size=0,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_sacle=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = shift_size

        assert 0 <= self.shift_size < self.window_size, \
            'the value of shift_size must bigger than 0 and smaller than window size'

        self.norm1 = norm_layer(dim)
        self.attention = WindowAttention(dim=dim,
                                         window_size=window_size,
                                         num_heads=num_heads,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_sacle,
                                         attn_drop=attn_drop,
                                         proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        # use layer scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, attn_mask):

        L = self.L
        B, N, C = x.shape

        shortcut = x
        x = self.norm1(x)

        pad_r = pad_l = 0
        if N % self.window_size != 0:
            pad_r = pad_l = (self.window_size - N % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_r, pad_l))
        _, Np, _ = x.shape

        # shift mask window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size,), dims=(1,))
        else:
            shifted_x = x
            attn_mask = None

        # windows partition
        # [B * num_windows, window_size, C]
        x_windows = window_partition(shifted_x.transpose(-1, -2), windows_size=self.window_size)

        # Window_Attention, Shift-Window Attention
        # [B * num_windows, window_size, C]
        attn_windows = self.attention(x_windows, attn_mask)
        # [B, C, Np]
        shifted_x = window_reverse(attn_windows, original_size=Np, window_size=self.window_size)
        # Transpose
        # [B, Np, C]
        shifted_x = shifted_x.transpose(-1, -2)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size,), dims=1)
        else:
            x = shifted_x

        # [B, Np, C] ==> [B, N, C]
        if pad_r > 0 or pad_l > 0:
            x = x[:, :N, :].contiguous()

        if not self.layer_scale:
            # ! 没有进行layer_scale
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = shortcut + self.drop_path(self.gamma1 * x)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))

        return x


class MCSwinlayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=8,
                 downsample=False,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_sacle=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = window_size // 2

        # build stages
        self.blocks = nn.Sequential(*[
            MCSwinTransformerBlock(dim=dim,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   shift_size=0 if (i % 2 == 0) else self.shift_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   qk_sacle=qk_sacle,
                                   drop=drop,
                                   attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   act_layer=act_layer,
                                   norm_layer=norm_layer,
                                   layer_scale=layer_scale)
            for i in range(depth)])
        self.downsample = downsample
        self.downsample_layer = ConvDownsampler(dim=dim) if downsample else nn.Identity()

    def create_mask(self, x, N):
        if N % self.window_size != 0:
            Np = int(np.ceil(N / self.window_size)) * self.window_size
        else:
            Np = N
        img_mask = torch.zeros((1, Np, 1), device=x.device)
        n_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for n in n_slices:
            img_mask[:, n, :] = cnt
            cnt += 1

        # [num_windows, window_size, 1]
        mask_windows = window_partition(img_mask.transpose(-1, -2), windows_size=self.window_size)
        # [num_windows, window_size]
        mask_windows = mask_windows.view(-1, self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, N):
        attn_mask = self.create_mask(x, N)
        for block in self.blocks:
            block.L = N
            x = block(x, attn_mask)
        x = self.downsample_layer(x)
        if self.downsample:
            N = (N + 1) // 2
        return x, N


class MCSwin_T(nn.Module):
    def __init__(self,
                 in_c,
                 num_cls,
                 h_args,
                 kernel_sizes,
                 strides,
                 out_channels,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 downscale=False,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=0.5):
        super().__init__()

        self.conv_embedding = Convolutional_Embedding(in_c=in_c,
                                                      kernel_sizes=kernel_sizes,
                                                      strides=strides,
                                                      out_channels=out_channels)

        self.patch_embedding = PatchEmerging(in_channels=out_channels[-1], out_channels=dim,
                                             patch_size=8, stride=8)

        self.SwinTransformerBlock = MCSwinlayer(dim=dim,
                                                depth=depth,
                                                num_heads=num_heads,
                                                window_size=window_size,
                                                downsample=downscale,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_sacle=qk_scale,
                                                drop=drop,
                                                attn_drop=attn_drop,
                                                act_layer=act_layer,
                                                norm_layer=norm_layer,
                                                layer_scale=layer_scale)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.ModuleList()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_channels = dim
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

    def forward(self, x):
        b = x.shape[0]
        x = self.conv_embedding(x)
        x = self.patch_embedding(x).transpose(-1, -2)
        _, N, _ = x.shape
        x, N = self.SwinTransformerBlock(x, N)
        x = self.avg_pool(x.transpose(-1, -2))
        x = x.view(b, -1)
        for module in self.classifier:
            x = module(x)
        return x


def mcswinT(in_c, h_args, num_cls):
    model = MCSwin_T(in_c=in_c, h_args=h_args, num_cls=num_cls,
                     kernel_sizes=[15, 9, 5, 3],
                     strides=[2, 1, 1, 1],
                     out_channels=[64, 128, 128, 192],
                     dim=128, depth=12, num_heads=8, window_size=32)
    return model

