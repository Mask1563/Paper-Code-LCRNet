import torch.nn as nn
import torch

 
is_elu = False

def activateELU(is_elu, nchan):
    if is_elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

def ConvBnActivate(in_channels, middle_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv3d(in_channels, middle_channels, stride=1, kernel_size=3, padding=1),
        nn.BatchNorm3d(middle_channels),
        activateELU(is_elu, middle_channels),

        nn.Conv3d(middle_channels, out_channels, stride=1, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        activateELU(is_elu, out_channels),
    )
    return conv

def DownSample():
    return nn.MaxPool3d(kernel_size=2, stride=2)

def UpSample(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

def FinalConvolution(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1)

def CatBlock(x1, x2):
    return torch.cat((x1, x2), 1)

