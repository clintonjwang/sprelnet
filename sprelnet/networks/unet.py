import math
import numpy as np
import torch
nn = torch.nn
F = nn.functional

def get_unet(net_HPs, dataset):
    return UNet(len(dataset["train label counts"])).cuda()

class MiniSeg(nn.Module):
    def __init__(self, channels_by_depth, kernels_by_depth, pool_depths=tuple()):
        super().__init__()
        pad_by_depth = [k//2 for k in kernels_by_depth]
        n_layers = len(kernels_by_depth)
        assert n_layers == len(channels_by_depth) - 1, "#channels and #kernels misaligned"

        self.layers = []
        for d in range(n_layers-1):
            self.layers += [nn.Conv2d(channels_by_depth[d], channels_by_depth[d+1],
                    kernel_size=kernels_by_depth[d], padding=pad_by_depth[d]),
                    nn.BatchNorm2d(channels_by_depth[d+1]), nn.ReLU()]
            if d in pool_depths:
                self.layers.append(nn.MaxPool2d(2))
            elif n_layers-d-1 in pool_depths:
                self.layers.append(nn.Upsample(scale_factor=2))

        self.layers.append(nn.Conv2d(channels_by_depth[-2], channels_by_depth[-1],
                    kernel_size=kernels_by_depth[-1], padding=pad_by_depth[-1]))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, X):
        return self.layers(X)


class DenoisingAE(nn.Module):
    def __init__(self, n_channels, bilinear=True, unet=None):
        super().__init__()
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        if unet is None:
            downs = [Down(64, 128), Down(128, 256),
                    Down(256, 512), Down(512, 1024)]
        else:
            downs = [unet.down1, unet.down2, unet.down3, unet.down4]
        self.downs = nn.Sequential(*downs)
        self.up1 = Up(1024 // (2 if bilinear else 1), 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)

    def forward(self, x):
        x = self.inc(x)
        x = self.downs(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        out = self.outc(x)
        return out


class UNet(nn.Module):
    def __init__(self, n_classes, n_channels=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = UNetUp(1024, 512 // factor, bilinear)
        self.up2 = UNetUp(512, 256 // factor, bilinear)
        self.up3 = UNetUp(256, 128 // factor, bilinear)
        self.up4 = UNetUp(128, 64, bilinear)
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

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UNetUp(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


