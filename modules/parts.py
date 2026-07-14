"""
This file contains the individual "Lego bricks" used to build the neural
networks in `models.py`. A segmentation network like U-Net takes an image
in, shrinks it down while learning what's in it, then grows it back to full
size while deciding which pixels belong to which class (e.g. "worm" vs
"background"). Each class below is one reusable piece of that process.

This module defines the reusable layers used by the models in `models.py`:
    - ResidualBlock:            GroupNorm -> ReLU -> Conv, twice, with a skip connection.
    - DownsampleBlock:          Encoder stage (conv block + 2x2 max-pool).
    - UpsampleBlock:            Decoder stage (bilinear upsample + skip fusion).
    - AttentionGate:            Additive attention used to gate skip connections.
    - UpsampleBlockWithAttention: Decoder stage that attention-gates the skip
                                 connection before fusing it, instead of using it raw.

Design notes that apply throughout:
    - GroupNorm is used instead of BatchNorm, which is generally preferable for
      small batch sizes (common in 3D/volumetric or memory-constrained biological
      imaging pipelines) since GroupNorm's statistics don't depend on batch size.
    - Upsampling uses bilinear interpolation + conv rather than ConvTranspose2d,
      to avoid the checkerboard artifacts that transposed convolutions are prone to.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A small building block that processes an image with two convolutions
    (pattern-detecting filters), while also keeping a direct copy of the
    original input around to add back in at the end. That "shortcut" copy
    makes it easier for the network to learn, especially as networks get deep,
    because it always has an easy fallback of "just pass the input through"
    if the learned transformation isn't helping.

    Pre-activation residual block: (GroupNorm -> ReLU -> Conv3x3) x 2, plus a
    skip connection from input to output.

    "Pre-activation" means normalization/activation happen *before* the
    convolution in each sub-block (GroupNorm -> ReLU -> Conv), which tends to
    give better gradient flow than the classic Conv -> Norm -> ReLU ordering.

    If in_channels != out_channels, a 1x1 conv projects the residual so it can
    be added elementwise to the main path's output; otherwise the residual
    connection is just an identity passthrough.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # GroupNorm needs num_channels to be divisible by num_groups.
        # `in_channels // 2` is used as a heuristic to pick a reasonably large
        # group count while staying <= 32 groups (the common default for GroupNorm).
        # NOTE: this heuristic can silently produce an invalid num_groups if
        # in_channels is not evenly divisible by the resulting value (e.g. odd
        # channel counts) -- worth double-checking if you change `features` in
        # the UNet configs.
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
        # Project (or pass through) the input so it matches the main path's
        # output channel count before the elementwise addition below.
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        # Final ReLU is applied *after* the residual addition (post-activation
        # for the block as a whole), even though each internal conv uses
        # pre-activation ordering.
        return self.relu(out)


class DownsampleBlock(nn.Module):
    """
    One "shrinking" step on the way into the network. It looks at the image
    with some convolutions, then shrinks it to half its width and height.
    Shrinking lets later layers see a bigger-picture view of the image, at
    the cost of fine detail -- which is why we hang onto the detailed version
    ("skip") before shrinking, to hand back to the decoder later.

    One encoder stage: a conv block followed by 2x2 max-pooling.

    Returns both the pooled (downsampled) tensor to feed to the next stage,
    and the pre-pool feature map ("skip") to be reused later by the decoder
    at the matching resolution.
    """

    def __init__(self, in_channels, out_channels, is_input_block=False):
        super().__init__()
        if is_input_block:
            # The very first block operates directly on raw input pixels
            # (e.g. a single/few-channel microscopy image), so normalization
            # is skipped here -- GroupNorm on raw, un-normalized input
            # channels tends to be unnecessary/unhelpful, and this also
            # sidesteps any num_groups-vs-in_channels divisibility issues
            # when in_channels is very small (e.g. 1 or 3).
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
    """
    One "growing" step on the way out of the network, the mirror image of
    DownsampleBlock. It doubles the image back up in size, then combines
    that with the matching detailed version saved earlier by the encoder
    (the "skip" connection), so the network can recover fine detail that
    would otherwise have been lost while shrinking.

    One decoder stage: upsample the low-resolution feature map, fuse it with
    the corresponding encoder skip connection, then refine with a ResidualBlock.

    Skip fusion can be done two ways (controlled by `concatenate_features`):
        - concatenate (channel-wise cat): preserves all information from both
          paths but doubles channel count into the following ResidualBlock,
          increasing parameter count.
        - additive (elementwise sum): cheaper, forces the two paths into the
          same channel dimensionality, and behaves like a residual connection
          from the encoder into the decoder.
    """

    def __init__(self, in_channels, out_channels, concatenate_features=True):
        super().__init__()

        # old implementation, removed due to checkerboard artifacts
        # self.up = nn.ConvTranspose2d(
        #     in_channels, out_channels, kernel_size=2, stride=2
        # )
        # Replaced with bilinear interpolation + conv (below), which avoids
        # the uneven kernel-overlap pattern that produces checkerboard
        # artifacts with strided transposed convolutions.

        self.concatenate_features = concatenate_features

        # The ResidualBlock that follows fusion needs to know how many
        # channels it will receive: concatenation doubles channels (out_channels
        # from the upsampled path + out_channels from the skip, since pre_conv
        # below maps into out_channels), while addition keeps channels
        # unchanged after the sum.
        if concatenate_features:
            residual_in_channels = in_channels
        else:
            residual_in_channels = in_channels // 2

        # Projects the upsampled feature map to `out_channels` so it can be
        # combined (concat or sum) with the skip connection, which already
        # has `out_channels` channels from the corresponding encoder stage.
        self.pre_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding="same"
        )

        self.conv = ResidualBlock(residual_in_channels, out_channels)

    def forward(self, x, skip):
        # Spatial upsampling via bilinear interpolation rather than a learned
        # transposed convolution -- see note above about checkerboard artifacts.
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
    """
    A small learned "spotlight" that highlights the useful parts of a saved
    encoder image and dims the irrelevant parts (like background noise),
    before that image gets combined into the decoder. Instead of blindly
    reusing everything from earlier in the network, the model learns which
    regions actually matter for the current decoding step.

    Additive attention gate (Oktay et al., "Attention U-Net").

    Learns a soft, per-pixel relevance mask over the encoder skip connection
    `x`, conditioned on the decoder's current (gating) signal `g`. Intuitively:
    "given what the decoder currently thinks is going on (g), which spatial
    regions of this skip connection (x) are actually useful to attend to?"

    This lets the network suppress irrelevant/background regions in skip
    connections (e.g. non-tissue background in a fluorescence image) before
    they're fused into the decoder, rather than passing every skip pixel
    through unfiltered as vanilla U-Net does.

    Args:
        F_g: number of channels in the gating signal `g` (decoder feature map).
        F_l: number of channels in the skip connection `x` (encoder feature map).
        F_int: number of intermediate channels used to compute the attention map;
            typically a bottleneck (smaller than F_g/F_l) for efficiency.
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        num_groups_int = min(32, F_int // 2)

        # 1x1 convs project both inputs into a shared F_int-dimensional space
        # so they can be combined additively below, regardless of their
        # original channel counts.
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
        # Collapses the combined signal down to a single-channel attention
        # map and squashes it to [0, 1] with sigmoid, giving a per-pixel
        # "how much of x's signal should pass through" gate.
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Additive attention: combine gating and skip signals, then nonlinearity.
        psi = self.relu(g1 + x1)
        # Reduce to a single-channel [0, 1] attention map, broadcast over x's channels.
        psi = self.psi(psi)
        # Rescale the skip connection by the learned attention map -- this is
        # the actual "gating" step. Regions with psi ~ 0 are suppressed;
        # regions with psi ~ 1 pass through unchanged.
        return x * psi


class UpsampleBlockWithAttention(nn.Module):
    """
    Same "growing" decoder step as UpsampleBlock, but it shines the
    AttentionGate spotlight on the saved encoder image first, before
    combining it in. This is what makes the "Attention" version of the
    network different from the plain version.

    Same role as UpsampleBlock (upsample -> fuse with skip -> refine), but the
    skip connection is passed through an AttentionGate before fusion, using
    the upsampled decoder features as the gating signal.

    This is the decoder stage used by AttentionUNet in models.py, as opposed
    to the plain UpsampleBlock used by the vanilla UNet.
    """

    def __init__(self, in_channels, out_channels, concatenate_features=True):
        super().__init__()
        self.concatenate_features = concatenate_features

        # Both the gating signal (post pre_conv, post upsample decoder
        # features) and the skip connection have `out_channels` channels at
        # this point, so F_g == F_l == out_channels. F_int is halved as a
        # standard bottleneck to keep the gate lightweight.
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

        # Gate the skip connection using the (already upsampled/projected)
        # decoder features as context, *before* fusing -- this is the key
        # difference from plain UpsampleBlock, which fuses the raw skip as-is.
        skip = self.attention(x, skip)
        if self.concatenate_features:
            x = torch.cat([x, skip], dim=1)
        else:
            x = x + skip
        return self.conv(x)