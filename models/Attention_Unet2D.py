import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Depthwise_Conv2D, ConvBlock2D, Encoder2D, Decoder2D, Attention_Gate2D, initialize_weights

class Attention_Unet2D(nn.Module):
    def __init__(self, conv_type="regular", initializer="kaiming"):
        super().__init__()
        Conv2d = nn.Conv2d if conv_type == "regular" else Depthwise_Conv2D
        
        # Define all the submodules
        self.encoder1 = Encoder2D(3, 64, conv_type=conv_type)
        self.encoder2 = Encoder2D(64, 128, conv_type=conv_type)
        self.encoder3 = Encoder2D(128, 256, conv_type=conv_type)
        self.encoder4 = Encoder2D(256, 512, conv_type=conv_type)
        
        self.center_block = ConvBlock2D(512, 1024, conv_type=conv_type)
        
        self.decoder4 = Decoder2D(512, conv_type=conv_type)
        self.decoder3 = Decoder2D(256, conv_type=conv_type)
        self.decoder2 = Decoder2D(128, conv_type=conv_type)
        self.decoder1 = Decoder2D(64, conv_type=conv_type)
        
        self.output_end = nn.Sequential(Conv2d(64, 1, kernel_size=3,
                                               stride=1, padding=1),
                                        nn.BatchNorm2d(num_features=1),
                                        nn.Sigmoid())
        
        # Define the attenion gates in the levels of 4, 3, and 2
        self.attn_gate4 = Attention_Gate2D(512, 1024, 512, conv_type=conv_type)
        self.attn_gate3 = Attention_Gate2D(256, 512, 256, conv_type=conv_type)
        self.attn_gate2 = Attention_Gate2D(128, 256, 128, conv_type=conv_type)
        
    def forward(self, x, return_attns='False'):
        x1, x2 = self.encoder1(x)
        x2, x3 = self.encoder2(x2)
        x3, x4 = self.encoder3(x3)
        x4, x5 = self.encoder4(x4)
        d5 = self.center_block(x5)
        
        # Take the use of attention gates
        x4, attn4 = self.attn_gate4(x4, d5)
        d4 = self.decoder4(d5, x4)
        x3, attn3 = self.attn_gate3(x3, d4)
        d3 = self.decoder3(d4, x3)
        x2, attn2 = self.attn_gate2(x2, d3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)
        
        out = self.output_end(d1)
        
        if return_attns is True:
            return out, {'attn2': attn2, 'attn3': attn3, 'attn4':attn4}
        else:
            return out