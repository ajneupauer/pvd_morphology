import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        num_groups = min(32, in_channels // 2)

        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding="same"
            ),
        )

        num_groups_out = min(32, out_channels // 2)

        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups_out, out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding="same"
            ),
        )

        self.relu = nn.ReLU()

        # If the input and output channels are different, we need to adjust
        # the residual connection
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return self.relu(out)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_input_block=False):
        super().__init__()
        if is_input_block:
            # Special case for input block - skip normalization
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding="same"
                ),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding="same"
                ),
            )
        else:
            self.conv = ResidualBlock(in_channels, out_channels)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, concatenate_features=True):
        super().__init__()

        # old implementation, removed due to checkerboard artifacts
        # self.up = nn.ConvTranspose2d(
        #     in_channels, out_channels, kernel_size=2, stride=2
        # )

        self.concatenate_features = concatenate_features

        if concatenate_features:
            residual_in_channels = in_channels
        else:
            residual_in_channels = in_channels // 2

        self.pre_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding="same"
        )

        self.conv = ResidualBlock(residual_in_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )

        x = self.pre_conv(x)

        if self.concatenate_features:
            x = torch.cat([x, skip], dim=1)
        else:
            x = x + skip

        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        num_groups_int = min(32, F_int // 2)

        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.GroupNorm(num_groups_int, F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.GroupNorm(num_groups_int, F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpsampleBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, concatenate_features=True):
        super().__init__()
        self.concatenate_features = concatenate_features

        self.attention = AttentionGate(
            F_g=out_channels, F_l=out_channels, F_int=out_channels // 2
        )

        if concatenate_features:
            residual_in_channels = in_channels
        else:
            residual_in_channels = in_channels // 2

        self.pre_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding="same"
        )

        self.conv = ResidualBlock(residual_in_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )

        x = self.pre_conv(x)

        skip = self.attention(x, skip)
        if self.concatenate_features:
            x = torch.cat([x, skip], dim=1)
        else:
            x = x + skip
        return self.conv(x)
