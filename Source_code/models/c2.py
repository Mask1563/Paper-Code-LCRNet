import torch
from torch import nn
from torchsummary import summary
 
class ChannelModule(nn.Module):
    def __init__(self, chanel, ratio=16):
        super(ChannelModule, self).__init__()
        c = chanel
        # _, c, _, _ = inputs.size()
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.share_liner = nn.Sequential(
            nn.Linear(c, c // ratio),
            nn.ReLU(),
            nn.Linear(c // ratio, c)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, inputs):
        x = self.maxpool(inputs).view(inputs.size(0), -1)#nc
        maxout = self.share_liner(x).unsqueeze(2).unsqueeze(3).unsqueeze(4)#nchwd
        y = self.avgpool(inputs).view(inputs.size(0), -1)
        avgout = self.share_liner(y).unsqueeze(2).unsqueeze(3).unsqueeze(4)#nchwd
        return self.sigmoid(maxout + avgout)
 
 
class SpatialModule(nn.Module):
    def __init__(self):
        super(SpatialModule, self).__init__()
        self.maxpool = torch.max
        self.avgpool = torch.mean
        self.concat = torch.cat
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, inputs):
        maxout, _ = self.maxpool(inputs, dim=1, keepdim=True)#n1hw
        avgout = self.avgpool(inputs, dim=1, keepdim=True)#n1hw
        
        outs = self.concat([maxout, avgout], dim=1)#n2hw
        
        outs = self.conv(outs)#n1hw
        return self.sigmoid(outs)
 
 
class CBAM(nn.Module):
    def __init__(self, chanel):
        super(CBAM, self).__init__()
        self.channel_out = ChannelModule(chanel)
        self.spatial_out = SpatialModule()
 
    def forward(self, inputs):
        outs = self.channel_out(inputs) * inputs
        return self.spatial_out(outs) * outs