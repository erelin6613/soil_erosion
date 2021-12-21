import torch
import torch.nn as nn
import torch.nn.functional as F


default_unet_config = {
    "down_blocks": [64, 128, 256, 512, 1024],
    "up_blocks": [512, 256, 128, 64],
    "num_classes": 1
}


def double_conv_block(in_channels, out_channels, k_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=k_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

def down_conv_block(in_channels, out_channels, scale=2):
    return nn.Sequential(
        nn.MaxPool2d(scale),
        double_conv_block(in_channels, out_channels)
        )

def up_conv_block(in_channels, out_channels, k_size=3):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=k_size, stride=2),
        double_conv_block(in_channels, out_channels)
    )

def out_block(in_channels, out_channels, k_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=k_size)
    )


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, config_file=None):
        super.__init__()

        self.encoder = nn.Sequential(
            down_conv_block()
        )


    def _build_from_config(self, config_file):
        pass

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
