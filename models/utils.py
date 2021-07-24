import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Depthwise_Conv2D(nn.Module):
    """
    This module implements the depth-wise 2D convolution.
    """
    def __init__(self, in_chn, out_chn,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.ConvPerChannel = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding,
                                        groups=in_chn)
        self.OneByOne = nn.Conv2d(in_chn, out_chn, kernel_size=1)

    def forward(self, x):
        x = self.ConvPerChannel(x)
        out = self.OneByOne(x)

        return out

class CBLayer2D(nn.Module):
    """
    This module implements convolution & batch-normalization & leaky_relu activation function.
    """
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, conv_type="regular"):
        super().__init__()
        Conv2d = nn.Conv2d if conv_type == "regular" else Depthwise_Conv2D

        self.conv = Conv2d(in_chn, out_chn, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(num_features=out_chn)

    def forward(self, x):
        out = F.leaky_relu(self.bn(self.conv(x)), inplace=True)

        return out

class ConvBlock2D(nn.Module):
    """
    This module implements the two-convolution block used in U-net's encoder and decoder.
    """
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, conv_type="regular"):
        super().__init__()

        self.CBL1 = CBLayer2D(in_chn, out_chn, kernel_size, stride, padding, conv_type)
        self.CBL2 = CBLayer2D(out_chn, out_chn, kernel_size, stride, padding, conv_type)

    def forward(self, x):
        return self.CBL2(self.CBL1(x))

class Residual_ConvBlock2D(nn.Module):
    """
    This module implements a residual convolution block; the skip connection spans over two convolutional layers
    """
    def __init__(self, num_chn, kernel_size=3, stride=1, padding=1, conv_type="regular"):
        super().__init__()

        self.CBL1 = CBLayer2D(num_chn, num_chn, kernel_size, stride, padding, conv_type)
        self.CBL2 = CBLayer2D(num_chn, num_chn, kernel_size, stride, padding, conv_type)

    def forward(self, x):
        return x + self.CBL2(self.CBL1(x))

class Encoder2D(nn.Module):
    """
    This module implements the encoder part of U-net, which output the result of convblock2D and pooling.
    """
    def __init__(self, in_chn, out_chn, conv_type="regular"):
        super().__init__()
        self.convblock = ConvBlock2D(in_chn, out_chn, conv_type=conv_type)

    def forward(self, x):
        x1 = self.convblock(x)
        x2 = F.max_pool2d(x1, kernel_size=2)

        return x1, x2

class Residual_Encoder2D(nn.Module):
    def __init__(self, in_chn, out_chn, conv_type="regular"):
        super().__init__()

        self.CBL = CBLayer2D(in_chn, out_chn, kernel_size=3, stride=1, padding=1, conv_type=conv_type)
        self.convblock = Residual_ConvBlock2D(out_chn, conv_type=conv_type)

    def forward(self, x):
        x1 = self.convblock(self.CBL(x))
        x2 = F.max_pool2d(x1, kernel_size=2)

        return x1, x2

class Decoder2D(nn.Module):
    """
    This module implements the decoder part of U-net: start with the upsampling module
    and followed by a convblock2D.
    """
    def __init__(self, out_chn, conv_type="regular"):
        super().__init__()
        Conv2d = nn.Conv2d if conv_type == "regular" else Depthwise_Conv2D

        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                Conv2d(2*out_chn, out_chn, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=out_chn))
        self.convblock = ConvBlock2D(2*out_chn, out_chn, conv_type=conv_type)

    def forward(self, x1, x2):
        x1 = F.leaky_relu(self.up(x1))

        return self.convblock(torch.cat((x1, x2), dim=1))

    
    
class Residual_Decoder2D(nn.Module):
    def __init__(self, out_chn, conv_type="regular"):
        super().__init__()
        Conv2d = nn.Conv2d if conv_type == "regular" else Depthwise_Conv2D

        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                Conv2d(2*out_chn, out_chn, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=out_chn))
        self.CBL = CBLayer2D(2*out_chn, out_chn, kernel_size=3, stride=1, padding=1, conv_type=conv_type)
        self.convblock = Residual_ConvBlock2D(out_chn, conv_type=conv_type)

    def forward(self, x1, x2):
        x1 = F.leaky_relu(self.up(x1))

        return self.convblock(self.CBL(torch.cat((x1, x2), dim=1)))

    
    
class Attention_Gate2D(nn.Module):
    def __init__(self, feat_chn, gate_chn, inter_chn, ds_rate=2, conv_type="regular"):
        super().__init__()
        Conv2d = nn.Conv2d if conv_type == "regular" else Depthwise_Conv2D

        self.Theta = Conv2d(feat_chn, inter_chn, kernel_size=ds_rate, stride=ds_rate, padding=0)
        self.Phi = Conv2d(gate_chn, inter_chn, kernel_size=1, stride=1, padding=0)
        self.Psi = Conv2d(inter_chn, 1, kernel_size=1, stride=1, padding=0)

        '''self.W = nn.Sequential(Conv2d(feat_chn, feat_chn, kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm2d(num_features=feat_chn))'''

    def forward(self, x, g):
        input_size = x.size()
        theta_x = self.Theta(x)
        inter_spatial = theta_x.size()[2:]

        phi_g = F.interpolate(self.Phi(g), size=inter_spatial, mode='bilinear')
        f = F.leaky_relu(theta_x+phi_g)

        grid = torch.sigmoid(self.Psi(f))
        grid = F.interpolate(grid, size=input_size[2:], mode='bilinear')

        y = grid.expand_as(x) * x
        '''out = self.W(y)'''

        return y, grid



    
"""
3D versions of the modules for the construction of the 3D models
"""
class Depthwise_Conv3D(nn.Module):
    """
    This module implements the depth-wise 3D convolution.
    """
    def __init__(self, in_chn, out_chn,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.ConvPerChannel = nn.Conv3d(in_chn, in_chn, kernel_size=kernel_size,
                                       stride=stride, padding=padding, groups=in_chn)
        self.OneByOne = nn.Conv3d(in_chn, out_chn, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        return self.OneByOne(self.ConvPerChannel(x))
    

class CBLayer3D(nn.Module):
    """
    This module implements 3D version of convolution & batch-normalization & leaky_relu activation function.
    """
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, conv_type="regular"):
        super().__init__()
        Conv3d = nn.Conv3d if conv_type == "regular" else Depthwise_Conv3D

        self.conv = Conv3d(in_chn, out_chn, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(num_features=out_chn)

    def forward(self, x):
        out = F.leaky_relu(self.bn(self.conv(x)), inplace=True)

        return out
    
    
class ConvBlock3D(nn.Module):
    """
    This module implements 3D version of the two-convolution block used in U-net's encoder and decoder.
    """
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, conv_type="regular"):
        super().__init__()

        self.CBL1 = CBLayer3D(in_chn, out_chn, kernel_size, stride, padding, conv_type)
        self.CBL2 = CBLayer3D(out_chn, out_chn, kernel_size, stride, padding, conv_type)

    def forward(self, x):
        return self.CBL2(self.CBL1(x))
        
        
        
class Residual_ConvBlock3D(nn.Module):
    """
    This module implements a 3D version of residual convolution block; the skip connection spans over two convolutional layers
    """
    def __init__(self, num_chn, kernel_size=3, stride=1, padding=1, conv_type="regular"):
        super().__init__()

        self.CBL1 = CBLayer3D(num_chn, num_chn, kernel_size, stride, padding, conv_type)
        self.CBL2 = CBLayer3D(num_chn, num_chn, kernel_size, stride, padding, conv_type)

    def forward(self, x):
        return x + self.CBL2(self.CBL1(x))
    
    

class Encoder3D(nn.Module):
    """
    This module implements 3D version of the encoder part of U-net, which output the result of convblock2D and pooling.
    """
    def __init__(self, in_chn, out_chn, conv_type="regular"):
        super().__init__()
        self.convblock = ConvBlock3D(in_chn, out_chn, conv_type=conv_type)

    def forward(self, x):
        x1 = self.convblock(x)
        x2 = F.max_pool3d(x1, kernel_size=2)

        return x1, x2    

    
    
class Residual_Encoder3D(nn.Module):
    def __init__(self, in_chn, out_chn, conv_type="regular"):
        super().__init__()

        self.CBL = CBLayer3D(in_chn, out_chn, kernel_size=3, stride=1, padding=1, conv_type=conv_type)
        self.convblock = Residual_ConvBlock3D(out_chn, conv_type=conv_type)

    def forward(self, x):
        x1 = self.convblock(self.CBL(x))
        x2 = F.max_pool3d(x1, kernel_size=2)
        
        return x1, x2
    
    
    
class Decoder3D(nn.Module):
    """
    This module implements the 3d decoder part of U-net3d: start with the upsampling module
    and followed by a convblock3D.
    """
    def __init__(self, out_chn, conv_type="regular"):
        super().__init__()
        Conv3d = nn.Conv3d if conv_type == "regular" else Depthwise_Conv3D

        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                Conv3d(2*out_chn, out_chn, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm3d(num_features=out_chn))
        self.convblock = ConvBlock3D(2*out_chn, out_chn, conv_type=conv_type)

    def forward(self, x1, x2):
        x1 = F.leaky_relu(self.up(x1))

        return self.convblock(torch.cat((x1, x2), dim=1))
    

    
class Residual_Decoder3D(nn.Module):
    def __init__(self, out_chn, conv_type="regular"):
        super().__init__()
        Conv3d = nn.Conv3d if conv_type == "regular" else Depthwise_Conv3D

        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                Conv3d(2*out_chn, out_chn, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm3d(num_features=out_chn))
        self.CBL = CBLayer3D(2*out_chn, out_chn, kernel_size=3, stride=1, padding=1, conv_type=conv_type)
        self.convblock = Residual_ConvBlock3D(out_chn, conv_type=conv_type)

    def forward(self, x1, x2):
        x1 = F.leaky_relu(self.up(x1))

        return self.convblock(self.CBL(torch.cat((x1, x2), dim=1)))
    


class Attention_Gate3D(nn.Module):
    def __init__(self, feat_chn, gate_chn, inter_chn, ds_rate=2, conv_type="regular"):
        super().__init__()
        Conv3d = nn.Conv3d if conv_type == "regular" else Depthwise_Conv3D

        self.Theta = Conv3d(feat_chn, inter_chn, kernel_size=ds_rate, stride=ds_rate, padding=0)
        self.Phi = Conv3d(gate_chn, inter_chn, kernel_size=1, stride=1, padding=0)
        self.Psi = Conv3d(inter_chn, 1, kernel_size=1, stride=1, padding=0)

        '''self.W = nn.Sequential(Conv2d(feat_chn, feat_chn, kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm2d(num_features=feat_chn))'''

    def forward(self, x, g):
        input_size = x.size()
        theta_x = self.Theta(x)
        inter_spatial = theta_x.size()[2:]

        phi_g = F.interpolate(self.Phi(g), size=inter_spatial, mode='trilinear')
        f = F.leaky_relu(theta_x+phi_g)

        grid = torch.sigmoid(self.Psi(f))
        grid = F.interpolate(grid, size=input_size[2:], mode='trilinear')

        y = grid.expand_as(x) * x
        '''out = self.W(y)'''

        return y, grid
    
    

def initialize_weights(m, type = 'kaiming'):
    if type not in ["kaiming", "xavier", "normal"]:
        raise NotImplementedError("Type not implemented; Please choose the following: 'kaiming', 'xavier', 'normal'.")

    module_class = m.__class__.__name__
    if module_class.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif type == "kaiming":
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif type == "xavier":
        init.xavier_normal_(m.weight.data, gain=1)
    elif type == "normal":
        init.normal_(m.weight.data, 0.0, 0.02)
        
def return_params_num(model):
    
    return sum([p.numel() for p in model.parameters()])