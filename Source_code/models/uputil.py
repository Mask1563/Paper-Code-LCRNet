import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb 

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
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
    


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm3d(inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x): 
        residue = x 

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        # out += self.shortcut(residue)
        out += residue

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

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
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
    def __init__(self, in_ch, out_ch, num_block, scale=(2,2,2),bottleneck=False):
        super().__init__()
        self.scale = scale

        self.conv_ch = nn.Conv3d(in_ch, out_ch, kernel_size=1)

        if bottleneck:
            block = BottleneckBlock
        else:
            # default
            block = BasicBlock


        block_list = []
        block_list.append(block(out_ch, out_ch))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)
        self.aff = AFF(channels=in_ch//2)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale, mode='trilinear', align_corners=True)
        x1 = self.conv_ch(x1)
        out = self.aff(x1, x2)
        out = self.conv(out)
        return out

class up_end(nn.Module):
    def __init__(self, scale=(2,2,2)):
        super().__init__()
        self.scale = scale


    def forward(self, x1):
        out = F.interpolate(x1, scale_factor=self.scale, mode='trilinear', align_corners=True)
        return out

