import torch.nn as nn

class Depthwise_Conv2D(nn.Module):
    def __init__(self,in_chn,out_chn,
                 kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.ConvPerChannel = nn.Conv2d(in_chn,in_chn,kernel_size,stride,padding,
                                        groups=in_chn)
        self.OneByOne = nn.Conv2d(in_chn,out_chn,kernel_size=1)

    def forward(self, x):
        x = self.ConvPerChannel(x)
        x = self.OneByOne(x)

        return x