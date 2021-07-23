import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Depthwise_Conv2D, CBLayer2D,\
    Residual_ConvBlock2D, Residual_Encoder2D, Attention_Gate2D


class Residual_Attention_Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature extraction modules
        self.encoder1 = Residual_Encoder2D(1,8)
        self.encoder2 = Residual_Encoder2D(8,16)
        self.encoder3 = Residual_Encoder2D(16,32)
        self.encoder4 = Residual_Encoder2D(32,64)
        self.encoder5 = Residual_ConvBlock2D(64)

        # Attention gates for level 3 and 4
        self.Attn_gate3 = Attention_Gate2D(32,64,32,
                                           ds_rate=4)
        self.Attn_gate4 = Attention_Gate2D(64,64,64,
                                           ds_rate=2)



    def forward(self,x):
        _, x2 = self.encoder1(x)
        _, x3 = self.encoder2(x2)
        x3,x4 = self.encoder3(x3)
        x4,x5 = self.encoder4(x4)
        gating = self.encoder5(x5)
        bsize,_,w,h = gating.size()

        out3,grid3 = self.Attn_gate3(x3,gating)
        out4,grid4 = self.Attn_gate4(x4,gating)

        out3 = torch.sum(out3, dim=(2,3))/torch.sum(grid3, dim=(2,3))
        out4 = torch.sum(out4, dim=(2,3))/torch.sum(grid4, dim=(2,3))
        out5 = F.avg_pool2d(gating, kernel_size=(w,h)).view(bsize,-1)

        return out3, out4, out5, grid3, grid4