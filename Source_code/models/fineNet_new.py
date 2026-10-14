"""FineNet variant with ConvNeXt-style 3D blocks and optional deep supervision."""
import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_
from timm.layers import to_3tuple
from .uputil import up_block, up_end
from .c2 import CBAM


def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv3d(in_chs, out_chs, stride=2, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_chs),
        nn.PReLU(out_chs)
    )


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]


class ConvNeXtBlock3D(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm3d(dim, eps=1e-6)

        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)

        x = input + self.drop_path(x)
        return x


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

        self.norm1 = nn.BatchNorm3d(hidden_features)
        self.norm2 = nn.BatchNorm3d(out_features)

        self.apply(self._init_weights)

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


def meta_blocks(dim, index, layers,
                mlp_ratio=4.,
                act_layer=nn.GELU, 
                drop_rate=0., drop_path_rate=0.,
                layer_scale_init_value=1e-6, vit_num=1):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        
        if index >= 3 and layers[index] - block_idx <= vit_num:
            blocks.append(CBAM(dim))
        else:
            blocks.append(ConvNeXtBlock3D(
                dim, 
                drop_path=block_dpr,
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


class FineNet_New(nn.Module):
    def __init__(self, layers=[3, 2, 6, 4, 1, 1],
                 embed_dims=[8, 16, 32, 64, 128],
                 vit_num=1,
                 mlp_ratios=4,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 num_classes=3,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_deep_supervision=True):
        super().__init__()
        
        self.use_deep_supervision = use_deep_supervision

        self.pe0 = stem(5, embed_dims[0])

        self.mb1 = meta_blocks(embed_dims[0], 0, layers,
                               mlp_ratio=mlp_ratios,
                               act_layer=act_layer, 
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        
        self.pe1 = Embedding(
            patch_size=down_patch_size, stride=down_stride,
            padding=down_pad,
            in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.mb2 = meta_blocks(embed_dims[1], 1, layers,
                               mlp_ratio=mlp_ratios,
                               act_layer=act_layer,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        
        self.pe2 = Embedding(
            patch_size=down_patch_size, stride=down_stride,
            padding=down_pad,
            in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.mb3 = meta_blocks(embed_dims[2], 2, layers,
                               mlp_ratio=mlp_ratios,
                               act_layer=act_layer,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        
        self.pe3 = Embedding(
            patch_size=down_patch_size, stride=down_stride,
            padding=down_pad,
            in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )
        self.mb4 = meta_blocks(embed_dims[3], 3, layers,
                               mlp_ratio=mlp_ratios,
                               act_layer=act_layer,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        
        self.pe4 = Embedding(
            patch_size=down_patch_size, stride=down_stride,
            padding=down_pad,
            in_chans=embed_dims[3], embed_dim=embed_dims[4]
        )
        self.mb5 = meta_blocks(embed_dims[4], 4, layers,
                               mlp_ratio=mlp_ratios,
                               act_layer=act_layer,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        self.mb6 = meta_blocks(embed_dims[4], 5, layers,
                               mlp_ratio=mlp_ratios,
                               act_layer=act_layer,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               layer_scale_init_value=layer_scale_init_value,
                               vit_num=vit_num)
        
        self.up1 = up_block(embed_dims[4], embed_dims[3], scale=(2, 2, 2), num_block=2)
        self.up2 = up_block(embed_dims[3], embed_dims[2], scale=(2, 2, 2), num_block=2)
        self.up3 = up_block(embed_dims[2], embed_dims[1], scale=(2, 2, 2), num_block=2)
        self.up4 = up_block(embed_dims[1], embed_dims[0], scale=(2, 2, 2), num_block=2)
        self.up5 = up_end()

        self.outc = nn.Conv3d(embed_dims[0], num_classes, kernel_size=1, bias=True)

        if use_deep_supervision:
            self.out_deep1 = nn.Conv3d(embed_dims[1], num_classes, kernel_size=1, bias=True)  # 1/4 resolution
            self.out_deep2 = nn.Conv3d(embed_dims[2], num_classes, kernel_size=1, bias=True)  # 1/8 resolution

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
        out_d2 = self.up2(out, x3)  # 1/8 resolution features
        out_d1 = self.up3(out_d2, x2)  # 1/4 resolution features
        out_d0 = self.up4(out_d1, x1)  # 1/2 resolution features
        out_final = self.up5(out_d0)

        if self.training and self.use_deep_supervision:
            final_out = self.outc(out_final)
            deep1_out = self.out_deep1(out_d1)
            deep2_out = self.out_deep2(out_d2)

            deep1_out = nn.functional.interpolate(deep1_out, size=final_out.shape[2:], 
                                                   mode='trilinear', align_corners=False)
            deep2_out = nn.functional.interpolate(deep2_out, size=final_out.shape[2:], 
                                                   mode='trilinear', align_corners=False)
            return [torch.sigmoid(final_out), torch.sigmoid(deep1_out), torch.sigmoid(deep2_out)]
        else:
            return torch.sigmoid(self.outc(out_final))
