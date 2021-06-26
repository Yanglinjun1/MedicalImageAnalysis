import torch.nn as nn
from .utils import Depthwise_Conv2D, ConvBlock2D, \
    Residual_Encoder2D, Residual_Decoder2D, initialize_weights

class Residual_Unet2D(nn.Module):
    def __init__(self, conv_type="regular", initializer="kaiming"):
        super().__init__()
        Conv2d = nn.Conv2d if conv_type == "regular" else Depthwise_Conv2D

        self.encoder1 = Residual_Encoder2D(in_chn=3, out_chn=64, conv_type=conv_type)
        self.encoder2 = Residual_Encoder2D(in_chn=64, out_chn=128, conv_type=conv_type)
        self.encoder3 = Residual_Encoder2D(in_chn=128, out_chn=256, conv_type=conv_type)
        self.encoder4 = Residual_Encoder2D(in_chn=256, out_chn=512, conv_type=conv_type)

        self.centerblock = ConvBlock2D(512, 1024, conv_type=conv_type)

        self.decoder4 = Residual_Decoder2D(512, conv_type=conv_type)
        self.decoder3 = Residual_Decoder2D(256, conv_type=conv_type)
        self.decoder2 = Residual_Decoder2D(128, conv_type=conv_type)
        self.decoder1 = Residual_Decoder2D(64, conv_type=conv_type)

        self.output_end = nn.Sequential(Conv2d(64, 1, kernel_size=3,
                                               stride=1, padding=1),
                                        nn.BatchNorm2d(num_features=1),
                                        nn.Sigmoid())

        if initializer:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                    initialize_weights(m, initializer)

    def forward(self, x):
        x1, x2 = self.encoder1(x)
        x2, x3 = self.encoder2(x2)
        x3, x4 = self.encoder3(x3)
        x4, x5 = self.encoder4(x4)

        d5 = self.centerblock(x5)
        d4 = self.decoder4(d5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)

        return self.output_end(d1)