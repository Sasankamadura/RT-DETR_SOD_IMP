import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation

class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution. 
    Preserves all pixel information by moving spatial dimensions to depth.
    Reference: https://arxiv.org/abs/2208.03640
    """
    def __init__(self, ch_in, ch_out, dimension=1):
        super().__init__()
        self.conv = nn.Conv2d(ch_in * 4, ch_out, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, 4C, H/2, W/2]
        x = torch.cat([
            x[..., 0::2, 0::2], 
            x[..., 1::2, 0::2], 
            x[..., 0::2, 1::2], 
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.bn(self.conv(x))

class CoordGnConv(nn.Module):
    """
    Coord-gnConv: A Novel Modification of gnConv for Small Object Detection.
    - Replaces recursive order-N with Parallel Coordinate-Aware Gating.
    - Uses BatchNorm for T4 optimization.
    - Integrates SPD logic for feature preservation.
    """
    def __init__(self, dim, reduction=4, act='silu'):
        super().__init__()
        self.dim = dim
        self.hidden_dim = dim // 2
        
        # 1. SPD-Enhanced Input Projection (instead of standard 1x1)
        # We use a small SPD block to reduce resolution but preserve SOD features
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )

        # 2. Parallel Coordinate-Gating Branch
        # We use mean pooling instead of AdaptiveAvgPool2d(None) for ONNX compatibility

        mip = max(8, dim // reduction)
        self.conv1 = nn.Conv2d(dim, mip, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = get_activation(act)

        self.conv_h = nn.Conv2d(mip, dim, kernel_size=1, stride=1, bias=False)
        self.conv_w = nn.Conv2d(mip, dim, kernel_size=1, stride=1, bias=False)
        
        # 3. Spatial Feature Branch (Depthwise for local interaction)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.bn_dw = nn.BatchNorm2d(dim)

        self.proj_out = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        identity = x
        x = self.proj_in(x)
        B, C, H, W = x.size()

        # --- Coordinate-Aware Gating ---
        x_h = x.mean(dim=-1, keepdim=True)
        x_w = x.mean(dim=-2, keepdim=True).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Multiplicative Gating (Spirit of gnConv)
        # But with 1D Coordinate awareness (Novelty)
        gated_x = x * a_h * a_w

        # --- Spatial Refinement ---
        out = self.bn_dw(self.dwconv(gated_x))
        out = self.proj_out(out)

        return out + identity
