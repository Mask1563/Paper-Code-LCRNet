import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb 
import math

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):

        super().__init__()

        if not mid_channels:

            mid_channels = out_channels

        self.double_conv = nn.Sequential(

            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),

            nn.BatchNorm2d(mid_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )



    def forward(self, x):

        return self.double_conv(x)





class Down(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.maxpool_conv = nn.Sequential(

            nn.MaxPool2d(2),

            DoubleConv(in_channels, out_channels)

        )



    def forward(self, x):

        return self.maxpool_conv(x)





class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):

        super().__init__()

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        else:

            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)

            self.conv = DoubleConv(in_channels, out_channels)





    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]

        diffX = x2.size()[3] - x1.size()[3]



        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,

                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)





class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)



    def forward(self, x):

        return self.conv(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
    


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x): 
        residue = x 

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)

        return out 

class BottleneckBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes//4, stride=1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes//4, planes//4, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes//4)

        self.conv3 = conv1x1(planes//4, planes, stride=1)
        self.bn3 = nn.BatchNorm2d(planes//4)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += self.shortcut(residue)

        return out



class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck:
            self.conv2 = BottleneckBlock(out_ch, out_ch)
        else:
            self.conv2 = BasicBlock(out_ch, out_ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, scale, num_block, bottleneck=False, pool=True):
        super().__init__()

        block_list = []

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock


        if pool:
            block_list.append(nn.MaxPool2d(scale))
            block_list.append(block(in_ch, out_ch))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, stride=1))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x):
        return self.conv(x)


class up_trans(nn.Module):
    def __init__(self,in_ch, out_ch, num_block):
        super().__init__()
        block_list = []
        block_list.append(BasicBlock(in_ch, out_ch * 2))

        for i in range(num_block - 1):
            block_list.append(BasicBlock(out_ch * 2, out_ch))
        self.conv = nn.Sequential(*block_list)

    def forward(self, x):
        out = self.conv(x)
        return out


class AFF(nn.Module):
    """
    多特征融合 AFF
    """

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        xa = x1 + x2
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x1 * wei + 2 * x2 * (1 - wei)
        return xo


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, scale=(2,2),bottleneck=False):
        super().__init__()
        self.scale = scale

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock


        block_list = []
        block_list.append(block(out_ch, out_ch))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)
        self.aff = AFF(channels=in_ch//2)

    def forward(self, x1, x2):
        # 根据给定的size或scale_factor参数来对输入进行下/上采样，使用的插值算法取决于参数mode的设置。
        # scale指定输出是输入的多少倍
        # align_corners如果设置为True，则输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值。如果设置为False，则输入和输出张量由它们的角像素的角点对齐，插值使用边界外值的边值填充; 当scale_factor保持不变时，使该操作独立于输入大小。
        # print(x1.shape)
        # print('上采样前后')
        x1 = F.interpolate(x1, scale_factor=self.scale, mode='bilinear', align_corners=True)
        # print(x1.shape)
        x1 = self.conv_ch(x1)
        # 先hadamard然后sum
        # out = torch.cat([x2, x1], dim=1)
        # AFF
        # print("x1:%s", x1.shape)
        # print("x2:%s", x2.shape)
        out = self.aff(x1, x2)
        # print("out:%s", out.shape)
        out = self.conv(out)
        # print(out.shape)
        return out

class up_end(nn.Module):
    def __init__(self, scale=(2,2)):
        super().__init__()
        self.scale = scale


    def forward(self, x1):
        # 根据给定的size或scale_factor参数来对输入进行下/上采样，使用的插值算法取决于参数mode的设置。
        # scale指定输出是输入的多少倍
        # align_corners如果设置为True，则输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值。如果设置为False，则输入和输出张量由它们的角像素的角点对齐，插值使用边界外值的边值填充; 当scale_factor保持不变时，使该操作独立于输入大小。

        out = F.interpolate(x1, scale_factor=self.scale, mode='bilinear', align_corners=True)
        # print(out.shape)
        return out

