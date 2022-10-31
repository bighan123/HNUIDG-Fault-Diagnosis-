import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

'''
The pytorch version of Vision Transformer
Orignal Author:
Github:

'''


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    1D data Patch Embedding
    """

    def __init__(self, data_size, in_c, patch_size, norm_layer):
        super().__init__()
        self.data_size = data_size
        self.patch_size = patch_size
        self.grid_size = data_size // patch_size
        self.num_patches = self.grid_size
        self.embed_dim = in_c * patch_size
        self.projection = nn.Conv1d(in_c, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, N = x.shape
        assert N == self.data_size, "the Input data size {} does not math the model {}".format(N, self.data_size)

        # [B, C, N] ==> [B, embed_dim, grid_size]
        x = self.projection(x)
        # [B,embed_dim, grid_size] ==> [B, grid_size, embed_dim]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() if act_layer else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim) if norm_layer else nn.Identity()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) if norm_layer else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    cfgs = {'base': [12, 8, 16],
            "middle_1": [12, 16, 16],
            'middle_2': [12, 16, 32],
            'large_1': [24, 16, 16],
            'large_2': [24, 16, 32]}

    def __init__(self, data_size, in_c, num_cls,
                 cfgs, h_args, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 norm_layer=None, act_layer=None, use_init=True):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_cls = num_cls
        depth, num_heads, patch_size = cfgs[0], cfgs[1], cfgs[2]
        self.last_channel = self.embed_dim = patch_size * in_c
        self.num_tokens = 1
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act = act_layer or nn.GELU
        self.patch_embed = PatchEmbed(data_size=data_size,
                                      patch_size=patch_size,
                                      in_c=in_c,
                                      norm_layer=norm_layer)
        self.num_patches = data_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()
        self.classifier = nn.ModuleList([])
        if not h_args:
            self.classifier.append(nn.Linear(self.last_channel, self.num_cls))
            self.classifier.append((nn.Softmax(dim=-1)))
        else:
            for i in range(len(h_args)):
                if i == 0:
                    self.classifier.append(nn.Linear(self.last_channel, h_args[i]))
                else:
                    self.classifier.append(nn.Linear(h_args[i - 1], h_args[i]))
            self.classifier.append(nn.Linear(h_args[-1], num_cls))
            self.classifier.append(nn.Softmax(dim=-1))
        if use_init:
            self._initialize_weights()

    def forward_features(self, x):
        # taking example: data: [B, 3, 1024], patch_size:16
        # [B, 3, 1024] ==> [B, 64, 48]
        x = self.patch_embed(x)
        # cls_token: [1, 1, 48] ==> [B, 1, 48]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # concat [B, 64, 48] abd [B, 1, 48] ==> [B, 65, 48]
        x = torch.cat((cls_token, x), dim=1)
        # x + position encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, _ = self.forward_features(x)
        for module in self.classifier:
            x = module(x)
        return x

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


cfgs = {'base': [12, 8, 16],
        "middle_1": [12, 16, 16],
        'middle_2': [12, 16, 32],
        'large_1': [24, 16, 16],
        'large_2': [24, 16, 32]}


def vit_base(data_size, in_c, num_cls, h_args, cfgs=cfgs['base']):
    model = VisionTransformer(data_size=data_size,
                              in_c=in_c,
                              num_cls=num_cls,
                              h_args=h_args,
                              cfgs=cfgs)
    return model


def vit_middle_patch16(data_size, in_c, num_cls, h_args, cfgs=cfgs["middle_1"]):
    model = VisionTransformer(data_size=data_size,
                              in_c=in_c,
                              num_cls=num_cls,
                              h_args=h_args,
                              cfgs=cfgs)
    return model


def vit_middle_patch32(data_size, in_c, num_cls, h_args, cfgs=cfgs["middle_2"]):
    model = VisionTransformer(data_size=data_size,
                              in_c=in_c,
                              num_cls=num_cls,
                              h_args=h_args,
                              cfgs=cfgs)
    return model


if __name__ == "__main__":
    model = vit_middle_patch16(data_size=1024, in_c=3, num_cls=8, h_args=[256, 128, 64, 32])
    tensor = torch.rand(2, 3, 1024)
    print(model(tensor).shape)
