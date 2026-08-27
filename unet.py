"""
U-Net architecture for image colorization.
Takes a grayscale (L channel) image and predicts the color (a, b channels).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic building block: Conv/ConvTranspose -> BatchNorm -> Activation"""

    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class UNetColorizer(nn.Module):
    """
    U-Net for image colorization.
    Input:  L channel  (1, 256, 256) - grayscale
    Output: ab channels (2, 256, 256) - predicted color
    """

    def __init__(self, in_channels=1, out_channels=2, features=64):
        super().__init__()

        # ---------- Encoder (downsampling) ----------
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )  # 256 -> 128

        self.down1 = ConvBlock(features, features * 2, down=True)       # 128 -> 64
        self.down2 = ConvBlock(features * 2, features * 4, down=True)   # 64 -> 32
        self.down3 = ConvBlock(features * 4, features * 8, down=True)   # 32 -> 16
        self.down4 = ConvBlock(features * 8, features * 8, down=True)   # 16 -> 8
        self.down5 = ConvBlock(features * 8, features * 8, down=True)   # 8 -> 4
        self.down6 = ConvBlock(features * 8, features * 8, down=True)   # 4 -> 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(),
        )  # 2 -> 1

        # ---------- Decoder (upsampling) ----------
        self.up1 = ConvBlock(features * 8, features * 8, down=False, use_dropout=True)
        self.up2 = ConvBlock(features * 8 * 2, features * 8, down=False, use_dropout=True)
        self.up3 = ConvBlock(features * 8 * 2, features * 8, down=False, use_dropout=True)
        self.up4 = ConvBlock(features * 8 * 2, features * 8, down=False)
        self.up5 = ConvBlock(features * 8 * 2, features * 4, down=False)
        self.up6 = ConvBlock(features * 4 * 2, features * 2, down=False)
        self.up7 = ConvBlock(features * 2 * 2, features, down=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh(),  # output range: -1 to 1 (matches normalized ab channels)
        )

    def forward(self, x):
        # Encoder pass (saving outputs for skip connections)
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        # Decoder pass (using skip connections)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))

        return self.final_up(torch.cat([up7, d1], dim=1))
