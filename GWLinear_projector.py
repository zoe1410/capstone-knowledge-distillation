import torch
import torch.nn as nn
# =========================
# Bottleneck Block (ResNet-style)
# =========================
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels=96, bottleneck=192, out_channels=768):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, bottleneck, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck)

        self.conv2 = nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck)

        self.conv3 = nn.Conv2d(bottleneck, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x


# =========================
# CNN Projector (thay Linear)
# =========================
class GWLinearProjector(nn.Module):
    def __init__(self, in_dim=96, out_dim=768, drop_p=0.4):
        super().__init__()

        # Bottleneck projection
        self.proj = BottleneckBlock(
            in_channels=in_dim,
            bottleneck=in_dim * 2,
            out_channels=out_dim
        )

        # Residual shortcut (quan trọng)
        self.shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

        self.dropout = nn.Dropout2d(drop_p)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, h_s):
        """
        Input:
            h_s: [B, 96, 14, 14] hoặc [B, 196, 96]
        Output:
            [B, 196, 768]
        """

        # ===== reshape về dạng CNN =====
        if h_s.dim() == 3:
            B, N, C = h_s.shape  # [B,196,96]
            h_s = h_s.view(B, 14, 14, C).permute(0, 3, 1, 2)  # [B,96,14,14]
        else:
            B, C, H, W = h_s.shape  # [B,96,14,14]

        # ===== forward =====
        x = self.dropout(h_s)

        out = self.proj(x)           # [B,768,14,14]
        shortcut = self.shortcut(x)  # [B,768,14,14]

        out = out + shortcut         # residual

        # ===== reshape về token =====
        out = out.permute(0, 2, 3, 1).reshape(B, 196, 768)

        out = self.ln(out)

        return out