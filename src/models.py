"""U-Net variants for dense AGB regression.

- UNet:          one encoder. Used for single-sensor runs and early fusion (channels concatenated).
- MidFusionUNet: one encoder per sensor; features are concatenated at every scale
                 and decoded by a shared decoder. Lets each branch learn sensor-specific
                 statistics before fusion.
"""
import torch
import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, in_ch, widths):
        super().__init__()
        self.blocks = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.blocks.append(conv_block(c, w))
            c = w
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feats = []
        for i, b in enumerate(self.blocks):
            x = b(x if i == 0 else self.pool(x))
            feats.append(x)
        return feats  # high -> low resolution


class Decoder(nn.Module):
    def __init__(self, enc_widths, out_ch=1):
        super().__init__()
        self.ups, self.blocks = nn.ModuleList(), nn.ModuleList()
        c = enc_widths[-1]
        for w in reversed(enc_widths[:-1]):
            self.ups.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.blocks.append(conv_block(2 * w, w))
            c = w
        self.head = nn.Conv2d(c, out_ch, 1)

    def forward(self, feats):
        x = feats[-1]
        for up, block, skip in zip(self.ups, self.blocks, reversed(feats[:-1])):
            x = block(torch.cat([up(x), skip], 1))
        return self.head(x)


class UNet(nn.Module):
    def __init__(self, in_channels, base=32, depth=4):
        super().__init__()
        widths = [base * 2**i for i in range(depth + 1)]
        self.enc = Encoder(sum(in_channels), widths)
        self.dec = Decoder(widths)

    def forward(self, xs):
        return self.dec(self.enc(torch.cat(xs, 1)))


class MidFusionUNet(nn.Module):
    def __init__(self, in_channels, base=32, depth=4):
        super().__init__()
        widths = [base * 2**i for i in range(depth + 1)]
        self.encs = nn.ModuleList([Encoder(c, widths) for c in in_channels])
        n = len(in_channels)
        self.fuse = nn.ModuleList([nn.Conv2d(n * w, w, 1) for w in widths])
        self.dec = Decoder(widths)

    def forward(self, xs):
        per_sensor = [enc(x) for enc, x in zip(self.encs, xs)]
        fused = [f(torch.cat(level, 1)) for f, level in zip(self.fuse, zip(*per_sensor))]
        return self.dec(fused)


def build_model(fusion, in_channels, base=32, depth=4):
    if fusion == "mid" and len(in_channels) > 1:
        return MidFusionUNet(in_channels, base, depth)
    return UNet(in_channels, base, depth)
