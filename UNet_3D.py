# UNet_3D.py

"""
3D U-Net implementation in PyTorch
Paper URL: https://arxiv.org/abs/1606.06650
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolutional layers each followed by batch normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class Encoder(nn.Module):
    """
    The Encoder part of the U-Net consisting of ConvBlocks and MaxPool layers.
    """

    def __init__(self, in_channels, feature_channels):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        for out_channels in feature_channels:
            block = ConvBlock(in_channels, out_channels)
            self.blocks.append(block)
            in_channels = out_channels

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features, x

class Decoder(nn.Module):
    """
    The Decoder part of the U-Net consisting of UpConvolutions and ConvBlocks.
    """

    def __init__(self, feature_channels):
        super(Decoder, self).__init__()
        self.upconvs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        reversed_channels = feature_channels[::-1]
        for idx in range(len(reversed_channels) - 1):
            upconv = nn.ConvTranspose3d(
                reversed_channels[idx], reversed_channels[idx+1], kernel_size=2, stride=2
            )
            self.upconvs.append(upconv)
            block = ConvBlock(reversed_channels[idx], reversed_channels[idx+1])
            self.blocks.append(block)

    def forward(self, x, features):
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            x = torch.cat([x, features[-(idx+1)]], dim=1)
            x = self.blocks[idx](x)
        return x

class UNet3D(nn.Module):
    """
    The UNet3D architecture combining the Encoder, Bottleneck, and Decoder.
    """

    def __init__(self, in_channels=1, out_channels=35, feature_channels=[64, 128, 256, 512]):
        super(UNet3D, self).__init__()
        self.encoder = Encoder(in_channels, feature_channels)
        self.bottleneck = ConvBlock(feature_channels[-1], feature_channels[-1]*2)
        self.decoder = Decoder(feature_channels)
        self.final_conv = nn.Conv3d(feature_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        features, x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, features)
        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 16, 128, 128)
    out = model(x)
    print("Output shape:", out.shape)
