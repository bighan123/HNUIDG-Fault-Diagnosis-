import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.Signal.vit import Block, Mlp, DropPath
from models.backbone.Signal.uniformer import PatchEmbed_for_uniformer



class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_sclae=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_sclae or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.Softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # only use cls token for q
        # [B, 1, C] ==> [B, 1, self.num_heads, head_dim] ==> [B, self.num_heads, 1, head_dim]
        q = self.q(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # [B,self.num_heads, 1, N]
        attn = q * self.scale @ k.transpose(-1, -2)
        attn = self.Softmax(attn)
        attn = self.attn_drop(attn)
        # [B, self.num_heads, 1, head_dim] => [B, self.num_heads, head_dim, 1]
        value = (attn @ v).transpose(1, 2)
        value = value.reshape(B, 1, C)
        value = self.proj(value)
        value = self.proj_drop(value)
        return value

class CrossAttentionBlock(nn.Module):
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
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_sclae=qk_scale,
                                   attn_drop=attn_drop,
                                   proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
    def forward(self, x):
        x = x[:,0:1,...] + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MultiAttentionBlock(nn.Module):
    'The branches depend on the length of dims'
    def __init__(self,
                 dims,
                 patches,
                 depths,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()

        num_branches = len(dims)
        self.num_branches = num_branches

        self.attns = nn.ModuleList()
        for br in range(num_branches):
            attn = nn.Sequential(*[Block(
                dim=dims[br],
                num_heads=num_heads[br],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_ratio=drop,
                attn_drop_ratio=attn_drop,
                drop_path_ratio=drop_path if isinstance(drop_path, float) else drop_path[i],
                norm_layer=norm_layer
            ) for i in range(depths[br])])
            self.attns.append(attn)

        if len(self.attns) == 0:
            self.attns = None

        self.projs = nn.ModuleList()
        for br in range(num_branches):
            proj = nn.Sequential(*[norm_layer(dims[br]), act_layer(), nn.Linear(dims[br], dims[(br + 1) % num_branches])])
            self.projs.append(proj)

        self.cross_attns = nn.ModuleList()
        for br in range(num_branches):
            br_ = (br + 1) % num_branches
            num_head_cross = num_heads[br_]
            cross_attn = nn.Sequential(*[CrossAttentionBlock(
                dim=dims[br_],
                num_heads=num_head_cross,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[-1],
                norm_layer=norm_layer
            ) for _ in range(depths[-1])])

            self.cross_attns.append(cross_attn)

        self.revert_projs = nn.ModuleList()
        for br in range(num_branches):
            revert_proj = nn.Sequential(*[norm_layer(dims[(br+1) % num_branches]), act_layer(), nn.Linear(dims[(br+1) % num_branches], dims[br])])
            self.revert_projs.append(revert_proj)

    def forward(self, x):
        attns_x = [attn(x_) for x_, attn in zip(x, self.attns)]
        # use cls token only
        proj_cls_token = [proj(x_[:,0:1]) for x_, proj in zip(attns_x, self.projs)]
        cross_outs = []
        for br in range(self.num_branches):
            #[B, 1, C] + [B, N - 1, C] ==> [B, N, C]
            # 计算small patch cls token和big patch序列的交互
            # 计算big patch cls token和small patch序列的交互
            cross_token = torch.cat((proj_cls_token[br], attns_x[(br + 1) % self.num_branches][:,1:,...]), dim = 1)
            # cls token for interaction
            cross_token = self.cross_attns[br](cross_token)
            reverted_cross_token = self.revert_projs[br](cross_token[:,0:1,...])
            cross_token = torch.cat((reverted_cross_token, attns_x[br][:, 1:, ...]), dim = 1)
            cross_outs.append(cross_token)
        return cross_outs

def _compute_pathes(data_size, patches):
    return [data // p  for data, p in zip(data_size, patches)]

class CrossVisionTransformer(nn.Module):
    def __init__(self, data_size, in_c, num_cls, h_args,
                 patch_sizes=(8,16), dims=(64,128),
                 depths=([1,3,1],[1,3,1],[1,3,1]),
                 num_heads=(8,16),
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU):
        super().__init__()
        self.num_cls = num_cls
        if not isinstance(data_size, list):
            data_size = [data_size] * len(dims)
        self.data_size = data_size
        self.patch_sizes = patch_sizes
        num_patches = _compute_pathes(data_size, patch_sizes)
        self.num_branches = len(dims)

        #add cls token
        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, dims[i])) for i in range(self.num_branches)])
        # add patch embedding
        self.patch_embed = nn.ModuleList()
        self.position_embedding = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], dims[i], dtype=torch.float, requires_grad=True))
                                                    for i in range(self.num_branches)])
        for d, p, dim in zip(data_size, patch_sizes, dims):
            self.patch_embed.append(PatchEmbed_for_uniformer(data_size=d,patch_size=p, in_c=in_c, out_dim=dim, norm_layer=norm_layer))


        self.pos_drop = nn.Dropout(p=drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depths])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.blocks = nn.ModuleList()
        dpr_ptr = 0
        for idx, depth_cfg in enumerate(depths):
            '''
            0,[1,3,1]
            1,[1,3,1]
            2,[1,3,1]
            '''
            cur_depth = max(depth_cfg[:-1]) + depth_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + cur_depth]
            block = MultiAttentionBlock(patches=patch_sizes,
                                        dims=dims,
                                        depths=depth_cfg,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drop_rate,
                                        attn_drop=attn_drop_rate,
                                        drop_path=dpr_,
                                        norm_layer=norm_layer,
                                        act_layer=act_layer)
            dpr_ptr += cur_depth
            self.blocks.append(block)

        self.norm = nn.ModuleList([norm_layer(dims[i]) for i in range(self.num_branches)])

        for i in range(self.num_branches):
            if self.position_embedding[i].requires_grad:
                torch.nn.init.trunc_normal_(self.position_embedding[i],mean=0,std=0.01,a=-2,b=2)
            torch.nn.init.trunc_normal_(self.cls_token[i], mean=0, std=0.01)

        self._initialize_weights()

        self.classifier = nn.ModuleList()
        for i in range(self.num_branches):
            each_last_channel = dims[i]
            each_classifier = []
            if not h_args:
                each_classifier.append(nn.Linear(each_last_channel, num_cls))
                each_classifier.append(nn.Softmax(dim=-1))
            else:
                for i in range(len(h_args)):
                    if i == 0:
                        each_classifier.append(nn.Linear(each_last_channel, h_args[i]))
                    else:
                        each_classifier.append(nn.Linear(h_args[i - 1], h_args[i]))
                each_classifier.append(nn.Linear(h_args[-1], num_cls))
                each_classifier.append(nn.Softmax(dim=-1))
            self.classifier.append(nn.Sequential(*each_classifier))

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
    def forward_features(self, x):
        B, C, N = x.shape
        multi_out = []
        for i in range(self.num_branches):
            if N % self.patch_sizes[i] != 0:
                N_ = math.ceil(N // self.patch_sizes[i]) * self.patch_sizes[i]
                x_ = F.interpolate(x, size=(N_,))
            else:
                x_ = x
            bi = self.patch_embed[i](x_).transpose(-1,-2)
            cls_token = self.cls_token[i].expand(B, -1, -1)
            bi = torch.cat([bi, cls_token], dim=1)
            bi = bi + self.position_embedding[i]
            bi = self.pos_drop(bi)
            multi_out.append(bi)
        for block in self.blocks:
            multi_out = block(multi_out)

        multi_out = [self.norm[i](single_out) for i, single_out in enumerate(multi_out)]
        out = [single_out[:,0,...] for single_out in multi_out ]

        return out

    def forward(self, x):
        x = self.forward_features(x)
        logits = []
        for i, x_ in enumerate(x):
            each_logits = self.classifier[i](x_)
            logits.append(each_logits)
        logits = torch.mean(torch.stack(logits,dim=0),dim=0)
        return logits

def cross_vit_tiny(data_size, in_c, h_args, num_cls, **kwargs):
    model = CrossVisionTransformer(data_size=data_size,
                                   in_c=in_c,
                                   h_args=h_args,
                                   num_cls=num_cls,
                                   patch_sizes=[8,16],
                                   dims=[64,128],
                                   depths=[[1,4,0],[1,4,0],[1,4,0]],
                                   num_heads=[8,8],
                                   **kwargs)
    return model

def cross_vit_base(data_size, in_c, h_args, num_cls, **kwargs):
    model = CrossVisionTransformer(data_size=data_size,
                                   in_c=in_c,
                                   h_args=h_args,
                                   num_cls=num_cls,
                                   patch_sizes=[8,16],
                                   dims=[128,256],
                                   depths=[[1,4,0],[1,4,0],[1,4,0]],
                                   num_heads=[16,16],
                                   **kwargs)
    return model

def cross_vit_big(data_size, in_c, h_args, num_cls, **kwargs):
    model = CrossVisionTransformer(data_size=data_size,
                                   in_c=in_c,
                                   h_args=h_args,
                                   num_cls=num_cls,
                                   patch_sizes=[16,32],
                                   dims=[192,384],
                                   depths=[[1,5,0],[1,5,0],[1,5,0]],
                                   num_heads=[32,32],
                                   **kwargs)
    return model

if __name__ == "__main__":
    model = cross_vit_tiny(2048,2,None,8)
    x = torch.rand(2,2,2048)
    print(model(x).shape)