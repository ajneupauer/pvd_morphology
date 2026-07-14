"""
This file assembles the pieces from `parts.py` into two complete, ready-to-use
segmentation networks: a standard U-Net, and an Attention U-Net that adds a
learned "spotlight" mechanism (see AttentionGate in parts.py) to focus on
relevant image regions. Both take an image in and output a per-pixel
prediction, e.g. a mask marking which pixels belong to a worm/cell/structure
of interest.

Two variants are provided:
    - UNet:          vanilla encoder-decoder with skip connections.
    - AttentionUNet:  same skeleton, but each decoder stage attention-gates
                      its skip connection (see UpsampleBlockWithAttention /
                      AttentionGate in parts.py) before fusing it in.

Both models share the same overall topology:
    input -> [DownsampleBlock] x len(features) -> bottleneck (ResidualBlock)
          -> [UpsampleBlock(WithAttention)] x len(features) -> 1x1 final_conv -> output

Note the two `pool`/`sigmoid` attributes assigned in __init__ but not
referenced in forward() (self.pool, self.sigmoid is only used conditionally):
these are currently vestigial in UNet -- pooling actually happens inside each
DownsampleBlock, not via `self.pool` at the top level. Harmless, but worth
knowing if you're auditing the module for dead code.
"""

import sys
sys.path.append('./modules')
import torch.nn as nn

from parts import (
    DownsampleBlock,
    ResidualBlock,
    UpsampleBlock,
    UpsampleBlockWithAttention,
)


class UNet(nn.Module):
    """
    A standard U-Net: shrinks the input image down through several stages to
    learn increasingly abstract, big-picture features, then grows it back up
    to the original size while reusing detail saved from the shrinking side,
    ending in a per-pixel prediction (e.g. a segmentation mask).

    Args:
        in_channels: number of channels in the input image (e.g. 1 for
            single-channel fluorescence, 3 for RGB).
        out_channels: number of output channels / segmentation classes
            (e.g. 1 for binary foreground/background masks).
        features: encoder channel widths from shallowest to deepest stage.
            The decoder mirrors this in reverse. len(features) also sets the
            depth of the network (number of down/up-sampling stages), so the
            input spatial size must be divisible by 2**len(features).
        concatenate_features: if True, decoder stages fuse skip connections
            via channel-wise concatenation; if False, via elementwise
            addition. See UpsampleBlock in parts.py for the tradeoffs.
        use_logits: if True, forward() returns raw logits (pair with
            nn.BCEWithLogitsLoss / nn.CrossEntropyLoss for numerically stable
            training). If False, a sigmoid is applied and forward() returns
            probabilities in [0, 1] directly (only appropriate for
            single-class/binary output, and typically paired with
            nn.BCELoss on already-squashed values).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        features=[64, 128, 256, 512],
        concatenate_features=False,
        use_logits=True,
    ):
        super().__init__()
        self.concatenate_features = concatenate_features
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_logits = use_logits
        self.sigmoid = nn.Sigmoid()

        # Down part of UNet
        # Build one DownsampleBlock per entry in `features`, e.g. for
        # features=[64,128,256,512] with in_channels=1:
        #   1 -> 64 -> 128 -> 256 -> 512, each stage halving spatial resolution.
        for i, feature in enumerate(features):
            is_first = i == 0
            self.downs.append(
                DownsampleBlock(in_channels, feature, is_input_block=is_first)
            )
            in_channels = feature

        # Up part of UNet
        # Mirror the encoder in reverse. `feature * 2` as in_channels reflects
        # that each UpsampleBlock receives features from one level deeper in
        # the network (i.e. from the previous, coarser decoder stage or the
        # bottleneck), which has twice the channel width of the corresponding
        # encoder skip connection it needs to fuse with.
        for feature in reversed(features):
            self.ups.append(
                UpsampleBlock(
                    feature * 2,
                    feature,
                    concatenate_features=self.concatenate_features,
                )
            )

        # Bridges encoder and decoder at the lowest spatial resolution /
        # highest channel width, doubling channels one more time
        # (features[-1] -> features[-1]*2) to match what the first
        # UpsampleBlock above expects as input.
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)
        # 1x1 conv maps the final decoder feature map to the desired number
        # of output classes/channels without altering spatial resolution.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling: run through each encoder stage, stashing the
        # pre-pool feature map from each stage for later reuse by the decoder.
        for down in self.downs:
            x, skip = down(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)
        # Reverse so skip_connections[0] is the deepest/coarsest skip
        # (matching the first, coarsest UpsampleBlock), and
        # skip_connections[-1] is the shallowest (matching the last
        # UpsampleBlock, right before final_conv).
        skip_connections = skip_connections[::-1]

        # Upsampling: fuse each decoder stage with its matching-resolution
        # encoder skip connection, from coarsest to finest.
        for i, up in enumerate(self.ups):
            skip = skip_connections[i]
            x = up(x, skip)

        x = self.final_conv(x)

        if self.use_logits:
            # Raw logits -- expected by BCEWithLogitsLoss / CrossEntropyLoss.
            return x
        else:
            # Pre-squashed probabilities in [0, 1].
            return self.sigmoid(x)


class AttentionUNet(nn.Module):
    """
    The same shrink-then-grow U-Net design as the UNet class above, but with
    one added twist: on the way back up, the network learns to "spotlight"
    the most relevant parts of the detail it saved earlier, rather than
    reusing all of it indiscriminately. This often helps the model ignore
    irrelevant background and focus on the structures that actually matter.

    Identical topology and constructor signature to UNet, except each
    decoder stage uses UpsampleBlockWithAttention instead of UpsampleBlock:
    before fusing a skip connection into the decoder, it's first passed
    through an AttentionGate conditioned on the decoder's current features.
    This lets the network learn to suppress irrelevant regions of the skip
    connection (e.g. background/noise) rather than passing it through raw.

    Note the default use_logits=False here (opposite of UNet's default),
    so forward() returns sigmoid-activated probabilities in [0, 1] out of
    the box unless overridden.

    See UNet's docstring above for the meaning of each constructor argument
    -- they're identical between the two classes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        features=[64, 128, 256, 512],
        concatenate_features=False,
        use_logits=False,
    ):
        super().__init__()
        self.concatenate_features = concatenate_features
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_logits = use_logits
        self.sigmoid = nn.Sigmoid()

        # Down part of UNet -- identical structure to UNet's encoder.
        for i, feature in enumerate(features):
            is_first = i == 0
            self.downs.append(
                DownsampleBlock(in_channels, feature, is_input_block=is_first)
            )
            in_channels = feature

        # Up part of UNet -- same channel bookkeeping as UNet, but each stage
        # is attention-gated (UpsampleBlockWithAttention) rather than a plain
        # UpsampleBlock.
        for feature in reversed(features):
            self.ups.append(
                UpsampleBlockWithAttention(
                    feature * 2,
                    feature,
                    concatenate_features=self.concatenate_features,
                ),
            )

        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling -- identical to UNet.forward().
        for down in self.downs:
            x, skip = down(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling -- each `up` call internally attention-gates `skip`
        # before fusing it in (see UpsampleBlockWithAttention.forward in
        # parts.py).
        for i, up in enumerate(self.ups):
            skip = skip_connections[i]
            x = up(x, skip)

        x = self.final_conv(x)

        if self.use_logits:
            return x
        else:
            return self.sigmoid(x)