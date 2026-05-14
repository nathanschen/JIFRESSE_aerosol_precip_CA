from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalUNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, hidden_channels)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(hidden_channels, hidden_channels * 2)
        self.bottleneck = ConvBlock(hidden_channels * 2, hidden_channels * 4)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = ConvBlock(hidden_channels * 4 + hidden_channels, hidden_channels * 2)
        self.occ_head = nn.Conv2d(hidden_channels * 2, 1, kernel_size=1)
        self.amount_head = nn.Conv2d(hidden_channels * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(enc2)
        upsampled = self.up(bottleneck)
        merged = torch.cat([upsampled, enc1], dim=1)
        decoded = self.dec(merged)
        occurrence_logits = self.occ_head(decoded)
        conditional_amount = self.amount_head(decoded)
        return occurrence_logits, conditional_amount


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch_size: int, spatial_shape: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_shape
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class ConvLSTMUNet(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(input_channels=input_channels, hidden_channels=hidden_channels)
        self.head = TemporalUNet(in_channels=hidden_channels, hidden_channels=hidden_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, channels, height, width = x.shape
        state = self.cell.init_state(batch_size, (height, width), x.device)
        hidden = None
        for step in range(sequence_length):
            hidden, cell = self.cell(x[:, step], state)
            state = (hidden, cell)
        return self.head(hidden)
