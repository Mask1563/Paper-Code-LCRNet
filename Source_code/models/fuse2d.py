"""
EfficientFormer implementation
"""
import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict
import itertools
from einops import rearrange

from timm.layers import DropPath, trunc_normal_

from timm.layers import to_2tuple
from .fuse2d_utils import up_block, up_trans, up_end, CBAM



def sparse_atttion(scores, k=0):
    v, _ = torch.topk(scores, k)
    vk = v[:, :, -1].unsqueeze(2).expand_as(scores)
    mask_k = torch.lt(scores, vk)
    scores = scores.masked_fill(mask_k, -1e18)
    return scores


class Attention(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=16):
        super().__init__()
        self.num_heads = num_heads
        # 根号d
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.register_buffer('attention_biases', torch.zeros(num_heads, 49))
        self.register_buffer('attention_bias_idxs',
                             torch.ones(49, 49).long())

        self.attention_biases_seg = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs_seg',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases_seg[:, self.attention_bias_idxs_seg]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.attention_biases_seg[:, self.attention_bias_idxs_seg] if self.training else self.ab
        bias = torch.nn.functional.interpolate(bias.unsqueeze(0), size=(attn.size(-2), attn.size(-1)), mode='bicubic')
        attn = attn + bias

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU()
    )


class Embedding(nn.Module):

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Flat(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x


class Pooling(nn.Module):

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class LinearMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)

        x = self.norm1(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        x = self.norm2(x)

        x = self.drop(x)
        return x


class Meta3D(nn.Module):

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Attention(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LinearMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))

        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Meta4D(nn.Module):

    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()
        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(x))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class Reshape(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.emb = Embedding(patch_size=patch_size, stride=stride,
                             padding=padding,
                             in_chans=in_chans, embed_dim=embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.contiguous().view(B, hidden, h, w)
        x = self.emb(x)
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def trans(x):
    B, n_patch, hidden = x.size()
    h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    x = x.contiguous().view(B, hidden, h, w)
    return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = 8
        W = 8
        x = self.expand(x)
        B, L, C = x.shape
        # print(B, L, C)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


def meta_blocks(dim, index, layers,
                pool_size=3, mlp_ratio=4.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                drop_rate=.0, drop_path_rate=0.,
                use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if index >= 3 and layers[index] - block_idx <= vit_num:
            blocks.append(
                CBAM(gate_channels=dim, reduction_ratio=mlp_ratio, pool_types=['avg', 'max'], no_spatial=True))
        else:
            blocks.append(Meta4D(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))

    blocks = nn.Sequential(*blocks)
    return blocks


class Fuse2d(nn.Module):
    def __init__(self, layers=[3, 2, 6, 4, 1, 1],
                 embed_dims=[16, 32, 64, 128, 256],
                 vit_num=1,
                 mlp_ratios=4,
                 pool_size=3,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 num_classes=3,
                 down_patch_size=3, down_stride=2, down_pad=1,  # k = 3x3; s = 2; p = 1;
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 ):
        super().__init__()
        self.pe0 = stem(4, embed_dims[0])
        self.mb1 = meta_blocks(embed_dims[0], 0, layers,
                               pool_size=pool_size, mlp_ratio=mlp_ratios,
                               act_layer=act_layer, norm_layer=norm_layer,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               use_layer_scale=use_layer_scale,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        self.pe1 = Embedding(
            patch_size=down_patch_size, stride=down_stride,
            padding=down_pad,
            in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.mb2 = meta_blocks(embed_dims[1], 1, layers,
                               pool_size=pool_size, mlp_ratio=mlp_ratios,
                               act_layer=act_layer, norm_layer=norm_layer,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               use_layer_scale=use_layer_scale,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        self.pe2 = Embedding(
            patch_size=down_patch_size, stride=down_stride,
            padding=down_pad,
            in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.mb3 = meta_blocks(embed_dims[2], 2, layers,
                               pool_size=pool_size, mlp_ratio=mlp_ratios,
                               act_layer=act_layer, norm_layer=norm_layer,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               use_layer_scale=use_layer_scale,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        self.pe3 = Embedding(
            patch_size=down_patch_size, stride=down_stride,
            padding=down_pad,
            in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )
        self.mb4 = meta_blocks(embed_dims[3], 3, layers,
                               pool_size=pool_size, mlp_ratio=mlp_ratios,
                               act_layer=act_layer, norm_layer=norm_layer,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               use_layer_scale=use_layer_scale,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        self.pe4 = Embedding(
            patch_size=down_patch_size, stride=down_stride,
            padding=down_pad,
            in_chans=embed_dims[3], embed_dim=embed_dims[4]
        )
        self.mb5 = meta_blocks(embed_dims[4], 4, layers,
                               pool_size=pool_size, mlp_ratio=mlp_ratios,
                               act_layer=act_layer, norm_layer=norm_layer,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               use_layer_scale=use_layer_scale,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        self.mb6 = meta_blocks(embed_dims[4], 5, layers,
                               pool_size=pool_size, mlp_ratio=mlp_ratios,
                               act_layer=act_layer, norm_layer=norm_layer,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               use_layer_scale=use_layer_scale,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        self.up1 = up_block(embed_dims[4], embed_dims[3], scale=(2, 2), num_block=2)
        self.up2 = up_block(embed_dims[3], embed_dims[2], scale=(2, 2), num_block=2)
        self.up3 = up_block(embed_dims[2], embed_dims[1], scale=(2, 2), num_block=2)
        self.up4 = up_block(embed_dims[1], embed_dims[0], scale=(2, 2), num_block=2)
        self.up5 = up_end()
        self.outc = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x0 = self.pe0(x)
        x1 = self.mb1(x0)
        x2_ = self.pe1(x1)
        x2 = self.mb2(x2_)
        x3_ = self.pe2(x2)
        x3 = self.mb3(x3_)
        x4_ = self.pe3(x3)
        x4 = self.mb4(x4_)

        x5_ = self.pe4(x4)
        x5__ = self.mb5(x5_)
        x5 = self.mb6(x5__)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.up5(out)
        out = self.outc(out)
        return torch.sigmoid(out)
