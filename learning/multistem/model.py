# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution → BatchNorm → ReLU (two conv layers)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class UNetMulti(nn.Module):
    """
    Multi-output U-Net for spectrogram masking.

    Input:
        x: (batch_size, in_channels, freq_bins, time_frames)
        Typical: in_channels=1 (magnitude or log-magnitude)

    Output:
        masks: (batch_size, n_outputs, freq_bins, time_frames)
               each channel in [0,1] after sigmoid (per-source mask)
    """

    def __init__(self, in_channels: int = 1, n_outputs: int = 4, base_channels: int = 64):
        """
        Args:
            in_channels: number of input channels (usually 1 for magnitude)
            n_outputs: number of output masks (e.g., 4 for vocals,bass,drums,other)
            base_channels: number of filters in the first layer (scales by 2^depth)
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_outputs = n_outputs
        self.base_channels = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Output conv -> n_outputs masks
        self.out_conv = nn.Conv2d(base_channels, n_outputs, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (B, in_channels, F, T)
        returns masks: (B, n_outputs, F, T)
        """
        # Encoder
        e1 = self.enc1(x)                 # (B, C, F, T)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # Decoder + skip connections (concatenate)
        d4 = self.up4(b)
        # If shapes differ by 1 due to odd dims, crop or pad accordingly:
        if d4.shape[-2:] != e4.shape[-2:]:
            d4 = self._match_tensor(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = self._match_tensor(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = self._match_tensor(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = self._match_tensor(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)           # (B, n_outputs, F, T)
        masks = self.out_act(out)         # sigmoid -> [0,1]
        return masks

    @staticmethod
    def _match_tensor(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        If spatial dims differ (due to pooling/conv rounding), crop or pad x to match ref.
        We prefer cropping center region of x.
        """
        _, _, hf, wf = x.shape
        _, _, hr, wr = ref.shape
        # crop height/freq
        if hf > hr:
            start_h = (hf - hr) // 2
            x = x[:, :, start_h:start_h + hr, :]
            hf = hr
        elif hf < hr:
            # pad symmetric
            pad_h = hr - hf
            pad_left = pad_h // 2
            pad_right = pad_h - pad_left
            x = F.pad(x, (0, 0, pad_left, pad_right))

        # crop width/time
        if wf > wr:
            start_w = (wf - wr) // 2
            x = x[:, :, :, start_w:start_w + wr]
        elif wf < wr:
            pad_w = wr - wf
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = F.pad(x, (pad_left, pad_right, 0, 0))

        return x


# Backwards-compatible alias name
UNet = UNetMulti


if __name__ == "__main__":
    # quick shape sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetMulti(in_channels=1, n_outputs=4, base_channels=32).to(device)
    x = torch.randn(2, 1, 256, 128).to(device)  # (batch, channel, freq_bins, time_frames)
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)  # -> (2, 4, 256, 128)
