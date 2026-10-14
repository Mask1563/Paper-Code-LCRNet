import torch.nn as nn
from .util import DownSample, ConvBnActivate, UpSample, FinalConvolution, CatBlock
from timm.layers import DropPath, trunc_normal_
import torch
from .c2 import CBAM
# from timm.layers import helpers
from timm.layers import to_3tuple
from .uputil import up_block, up_end


def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv3d(in_chs, out_chs, stride=2, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_chs),
        nn.PReLU(out_chs)
    )


class Pooling(nn.Module):

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool3d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = nn.PReLU(hidden_features)
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

        self.norm1 = nn.BatchNorm3d(hidden_features)
        self.norm2 = nn.BatchNorm3d(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
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


class PoolBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()
        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
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
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(x))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
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
            blocks.append(CBAM(dim))
        else:
            blocks.append(PoolBlock(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))

    blocks = nn.Sequential(*blocks)
    return blocks


class Embedding(nn.Module):

    def __init__(self, patch_size=3, stride=3, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm3d):
        super().__init__()
        stride = to_3tuple(stride)
        padding = to_3tuple(padding)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class FineNet(nn.Module):
    def __init__(self, layers=[3, 2, 6, 4, 1, 1],
                 embed_dims=[8, 16, 32, 64, 128],
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
        self.pe0 = stem(5, embed_dims[0])
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
        self.up1 = up_block(embed_dims[4], embed_dims[3], scale=(2, 2, 2), num_block=2)
        self.up2 = up_block(embed_dims[3], embed_dims[2], scale=(2, 2, 2), num_block=2)
        self.up3 = up_block(embed_dims[2], embed_dims[1], scale=(2, 2, 2), num_block=2)
        self.up4 = up_block(embed_dims[1], embed_dims[0], scale=(2, 2, 2), num_block=2)
        self.up5 = up_end()
        self.outc = nn.Conv3d(embed_dims[0], num_classes, kernel_size=1, bias=True)

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