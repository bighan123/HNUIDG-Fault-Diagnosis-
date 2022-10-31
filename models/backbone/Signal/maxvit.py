import math
import torch
import torch.nn as nn
from models.backbone.Signal.vit import Mlp
import torch.nn.functional as F
from models.backbone.Signal.vit import DropPath
'''
The pytorch version (deal with one dimension data)of Maxvit 
https://arxiv.org/abs/2204.01697
Original code: https://github.com/ChristophReich1996/MaxViT/blob/master/maxvit/maxvit.py
'''



class Depthwise_conv_3x3(nn.Module):
    def __init__(self, in_c,
                 out_channels,
                 downscale,
                 ):
        super().__init__()
        if not downscale:
            assert in_c == out_channels, "If downscaling is not utilized input and output channels must be equal"
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_channels=in_c, out_channels=out_channels,
                      kernel_size=1, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.GELU())
        self.Dconv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=2 if downscale else 1, padding=1, groups=out_channels)
        self.Pconv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.Dconv(x)
        x = self.Pconv(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_c, rd_ratio):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            nn.Linear(in_c, int(in_c * rd_ratio)),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(int(in_c * rd_ratio), in_c),
            nn.Sigmoid()
        )

    def forward(self, x):
        b = x.shape[0]
        v = x
        # [B, C, N] ==> [B, C, 1]
        x = self.avg_pool(x)
        # [B, C]
        x = x.view(b, -1)
        attn = self.layer2(self.layer1(x)).unsqueeze(-1)
        return attn * v


class MBconv(nn.Module):
    def __init__(self, in_c, out_channels,
                 downscale, drop_path=0.):
        super().__init__()
        if not downscale:
            assert in_c == out_channels, "If downscaling is not utilized input and output channels must be equal"
        self.drop_path = drop_path
        self.dbconv = Depthwise_conv_3x3(in_c=in_c,
                                         out_channels=out_channels,
                                         downscale=downscale)
        self.senet = SqueezeExcite(in_c=out_channels, rd_ratio=0.25)
        self.conv1x1 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1)
        if downscale:
            self.down_sample = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channels=in_c, out_channels=out_channels, kernel_size=1))
        else:
            self.down_sample = nn.Identity()

    def forward(self, x):
        short_cut = x
        x = self.dbconv(x)
        x = self.senet(x)
        x = self.conv1x1(x)
        if self.drop_path > 0:
            x = DropPath(x, drop_prob=self.drop_path)
        return x + self.down_sample(short_cut)


def window_partition(input, windows_size):
    '''

    :param input: (torch.Tensor), the shape should be [B, C, N]
    :param windows_size: (int) the one dim window size to be applied, Default(16)
    :return: Unfolded input tensor of the shape [B * window_nums, windows_size, C]
    '''
    B, C, N = input.shape
    # the window_nums should be N//windows_size
    windows = input.view(B, C, N // windows_size, windows_size)
    windows = windows.permute(0, 2, 3, 1).contiguous().view(-1, windows_size, C)
    return windows


def window_reverse(windows, original_size, window_size):
    '''
    :param windows: (torch.Tensor), the shape should be [B * window_nums, windows_size, C]
    :param original_size: the original shape
    :param window_size: the one dim window size to be applied, Default(16)
    :return: Folded output tensor of the shape [B, C, original_size]
    '''

    N = original_size
    B = int((windows.shape[0]) / (N / window_size))
    output = windows.view(B, N // window_size, window_size, -1)
    output = output.permute(0, 3, 1, 2).contiguous().view(B, -1, N)
    return output


def grid_partition(input, grid_size):
    '''
    :param input: (torch.Tensor), the shape should be [B, C, N]
    :param grid_size: (int) the one dim window size to be applied, Default(16)
    :return: Unfolded input tensor of the shape [B * window_nums, windows_size, C]
    '''

    B, C, N = input.shape
    grid = input.view(B, C, grid_size, N // grid_size)
    grid = grid.permute(0, 3, 2, 1).contiguous().view(-1, grid_size, C)
    return grid


def grid_reverse(grid, original_size, grid_size):
    N = original_size
    C = grid_size.shape[-1]
    B = int((grid_size.shape[0] / (N // grid_size)))
    output = grid.view(B, N // grid_size, grid_size, C)
    output = output.permute(0, 3, 1, 2).contiguous().view(B, C, N)
    return output


class RelativeSelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 grid_window_size,
                 attn_drop=0.0,
                 drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.grid_window_size = grid_window_size
        self.qkv_scale = num_heads ** -0.5
        self.attn_area = grid_window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)

        self.position_encoding = nn.Parameter(torch.zeros(1, num_heads, grid_window_size, grid_window_size))
        nn.init.trunc_normal_(self.position_encoding, mean=0, std=0.2)

    def forward(self, input):
        B, N, C = input.shape
        qkv = self.qkv(input)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # [B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.qkv_scale
        # [B, num_heads, N, head_dim] @ [B, num_heads, head_dim, N] = [B, num_heads, N, N]
        attn = q @ k.transpose(-2, -1) + self.position_encoding
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # [B, num_heads, N, head_dim] ==> [B, N, num_heads, head_dim] ==> [B, N, dim]
        output = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        output = self.proj(output)
        output = self.proj(output)
        return output


class MaxViTTransformerBlock(nn.Module):
    '''
            With block partition:
        x ← x + Unblock(RelAttention(Block(LN(x))))
        x ← x + MLP(LN(x))
        With grid partition:
        x ← x + Ungrid(RelAttention(Grid(LN(x))))
        x ← x + MLP(LN(x))
        Layer Normalization (LN) is applied after the grid/window partition to prevent multiple reshaping operations.
        Grid/window reverse (Unblock/Ungrid) is performed on the final output for the same reason.

    '''

    def __init__(self,
                 dim,
                 partition_function,
                 reverse_function,
                 num_heads=16,
                 grid_window_size=32,
                 attn_drop=0.0,
                 drop=0.0,
                 drop_path=0.0,
                 mlp_ratio=4):
        super().__init__()
        self.partition_function = partition_function
        self.reverse_function = reverse_function
        self.grid_window_size = grid_window_size
        self.norm_1 = nn.LayerNorm(dim)

        self.attention = RelativeSelfAttention(dim=dim,
                                               num_heads=num_heads,
                                               grid_window_size=grid_window_size,
                                               attn_drop=attn_drop,
                                               drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm_2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        B, C, N = x.shape
        assert N % self.grid_window_size == 0, 'the length must be an integer multiple of grid size or window size '
        # [B, C, N] ==> [B * grid_window_nums, grid_window_size, C]
        input_partitioned = self.partition_function(x, self.grid_window_size)
        short_cut = input_partitioned
        input_partitioned = self.norm_1(input_partitioned)
        input_partitioned = self.attention(input_partitioned)
        input_partitioned = self.drop_path(input_partitioned)
        out = input_partitioned + short_cut
        out = out + self.drop_path(self.mlp(self.norm_2(out)))
        # [B * grid_window_nums, grid_window_sie, C] ==> [B, C, N]
        out = self.reverse_function(out, N, self.grid_window_size)
        return out


class MasViTBlock(nn.Module):
    def __init__(self,
                 dim,
                 out_channels,
                 downscale=False,
                 num_heads=8,
                 grid_window_size=32,
                 attn_drop=0.,
                 drop=0.,
                 drop_path=0.,
                 mlp_ratio=4.):
        super().__init__()
        self.mb_conv = MBconv(in_c=dim,
                              out_channels=out_channels,
                              downscale=downscale, drop_path=drop_path)
        self.block_transformer = MaxViTTransformerBlock(dim=out_channels,
                                                        partition_function=window_partition,
                                                        reverse_function=window_reverse,
                                                        num_heads=num_heads,
                                                        grid_window_size=grid_window_size,
                                                        attn_drop=attn_drop,
                                                        drop=drop,
                                                        drop_path=drop_path,
                                                        mlp_ratio=mlp_ratio)
        self.grid_transformer = MaxViTTransformerBlock(dim=out_channels,
                                                       partition_function=window_partition,
                                                       reverse_function=window_reverse,
                                                       num_heads=num_heads,
                                                       grid_window_size=grid_window_size,
                                                       attn_drop=attn_drop,
                                                       drop=drop,
                                                       drop_path=drop_path,
                                                       mlp_ratio=mlp_ratio)

    def forward(self, x):
        x = self.mb_conv(x)
        x = self.block_transformer(x)
        x = self.grid_transformer(x)
        return x


class MaxVitStage(nn.Module):
    def __init__(self,
                 depth,
                 dim,
                 out_channels,
                 num_heads=8,
                 grid_window_size=32,
                 attn_drop=0.,
                 drop=0.,
                 drop_path=0.,
                 mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.Sequential(*[
            MasViTBlock(
                dim=dim if index == 0 else out_channels,
                out_channels=out_channels,
                downscale=index == 0,
                num_heads=num_heads,
                grid_window_size=grid_window_size,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                mlp_ratio=mlp_ratio) for index in range(depth)])

    def forward(self, x):
        out = self.blocks(x)
        return out


class MaxViT(nn.Module):
    '''
    Args:
        in_c: (int,optional): the input channel of input data
        depths: (Tuple[int, ....], optional): the Depth of each network stage
        channels:(Tuple[int, ....], optional): Number of channels in each network stage.

    '''

    def __init__(self,
                 in_c,
                 depths,
                 channels,
                 num_cls,
                 h_args=None,
                 dim=64,
                 num_heads=8,
                 grid_window_size=32,
                 attn_drop=0.,
                 drop=0.,
                 drop_path=0.,
                 mlp_ratio=4,
                 is_last=False):
        super().__init__()
        assert len(depths) == len(channels), 'For each stage a channel dimension must be given'
        self.num_cls = num_cls
        self.is_last = is_last
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=in_c,
                      out_channels=dim,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(True)
        )
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        self.stages = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            self.stages.append(
                MaxVitStage(
                    depth=depth,
                    dim=dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    num_heads=num_heads,
                    grid_window_size=grid_window_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=mlp_ratio
                )
            )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.ModuleList()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_channels = sum(channels) if self.is_last else channels[-1]
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
        x = self.stem(x)
        feature_list = []
        for stage in self.stages:
            x = stage(x)
            feature_list.append(x)
        if self.is_last:
            f_ = feature_list[0]
            for fea in feature_list[1:]:
                f_ = F.interpolate(f_, size=fea.shape[1])
                f_ = torch.cat([fea, f_], dim=1)
        else:
            f_ = x
        out = self.avg_pool(f_)
        out = out.view(b, -1)
        for module in self.classifier:
            out = module(out)
        return out


def max_vit_tiny_16(in_c, h_args, num_cls):
    model = MaxViT(
        depths=(2, 2, 5, 2),
        channels=(64, 128, 256, 512),
        dim=64,
        grid_window_size=16,
        in_c=in_c,
        h_args=h_args,
        num_cls=num_cls)
    return model


def max_vit_tiny_32(in_c, h_args, num_cls):
    model = MaxViT(
        depths=(2, 2, 5, 2),
        channels=(64, 128, 256, 512),
        dim=64,
        grid_window_size=32,
        in_c=in_c,
        h_args=h_args,
        num_cls=num_cls)
    return model


def max_vit_small_16(in_c, h_args, num_cls):
    model = MaxViT(
        depths=(2, 2, 5, 2),
        channels=(96, 128, 256, 512),
        dim=64,
        grid_window_size=16,
        in_c=in_c,
        h_args=h_args,
        num_cls=num_cls)
    return model


def max_vit_small_32(in_c, h_args, num_cls):
    model = MaxViT(
        depths=(2, 2, 5, 2),
        channels=(96, 128, 256, 512),
        dim=64,
        grid_window_size=32,
        in_c=in_c,
        h_args=h_args,
        num_cls=num_cls)
    return model


if __name__ == "__main__":
    net = max_vit_tiny_16(in_c=2, h_args=None, num_cls=8)
    tensor = torch.rand(8, 2, 1024)
    print(net(tensor).shape)
