import torch.nn as nn
from .utils import Depthwise_Conv2D, ConvBlock2D, Encoder2D, Decoder2D, initialize_weights

class Unet2D(nn.Module):
    def __init__(self, conv_type="regular", initializer="kaiming"):
        super().__init__()
        Conv2d = nn.Conv2d if conv_type == "regular" else Depthwise_Conv2D

        # Define all the sub-modules
        self.encoder1 = Encoder2D(3, 64, conv_type)
        self.encoder2 = Encoder2D(64, 128, conv_type)
        self.encoder3 = Encoder2D(128, 256, conv_type)
        self.encoder4 = Encoder2D(256, 512, conv_type)

        self.center_block = ConvBlock2D(512, 1024, conv_type)

        self.decoder4 = Decoder2D(512, conv_type)
        self.decoder3 = Decoder2D(256, conv_type)
        self.decoder2 = Decoder2D(128, conv_type)
        self.decoder1 = Decoder2D(64, conv_type)

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
        x5 = self.center_block(x5)

        d4 = self.decoder4(x5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)

        return self.output_end(d1)