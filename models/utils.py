import torch
import torch.nn as nn
import torch.nn.functional as F

class Depthwise_Conv2D(nn.Module):
    """
    This module implements the depth-wise 2D convolution
    """
    def __init__(self, in_chn, out_chn,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.ConvPerChannel = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding,
                                        groups=in_chn)
        self.OneByOne = nn.Conv2d(in_chn, out_chn, kernel_size=1)

    def forward(self, x):
        x = self.ConvPerChannel(x)
        x = self.OneByOne(x)

        return x


class ConvBlock2D(nn.Module):
    def __init__(self, in_chn, out_chn, conv_type = "regular"):
        super().__init__()
        Conv2d = nn.Conv2d if conv_type == "regular" else Depthwise_Conv2D

        self.conv1 = Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_chn)
        self.conv2 = Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_chn)

    def forward(self, x):
        x = F.leaky_relu((self.bn1(self.conv1(x))))
        out = F.leaky_relu((self.bn2(self.conv2(x))))

        return out

class Encoder2D(nn.Module):
    def __init__(self, in_chn, out_chn, conv_type="regular"):
        super().__init__()
        self.convblock = ConvBlock2D(in_chn, out_chn, conv_type)

    def forward(self, x):
        x1 = self.convblock(x)
        x2 = F.max_pool2d(x1, kernel_size=2)

        return x1, x2

class Decoder2D(nn.Module):
    def __init__(self, out_chn, conv_type="regular"):
        super().__init__()
        Conv2d = nn.Conv2d if conv_type == "regular" else Depthwise_Conv2D
        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                Conv2d(2*out_chn, out_chn, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=out_chn))
        self.convblock = ConvBlock2D(2*out_chn, out_chn, conv_type)

    def forward(self, x1, x2):
        x1 = F.leaky_relu(self.up(x1))

        return self.convblock(torch.cat((x1, x2), dim=1))