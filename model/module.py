import torch
import torch.nn as nn
import torch.nn.functional as F

class CLB(nn.Module):

    def __init__(self, input_dim: int):
        super(CLB, self).__init__()

        self.branch3 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=1),
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(0.1)
        )
    def forward(self,x):
        return self.branch3(x)


class ECAB(nn.Module):
    def __init__(self, input_dim: int):
        super(ECAB, self).__init__()
        # Example channel split. Suppose "C" is input_dim
        # (You might need to adjust the dimensions based on your exact architecture.)
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, kernel_size=1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, kernel_size=1),
            nn.Conv2d(input_dim // 2, input_dim // 2, kernel_size=3, padding=1),
            nn.Conv2d(input_dim // 2, input_dim // 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        # Concatenate along the channel dimension
        out = torch.cat([out1, out2], dim=1)

        return out

class ETB(nn.Module):
    def __init__(self, input_dim: int):
        super(ETB, self).__init__()

        # Branch 1 => (B, C//2, H, W), with attention inside
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, kernel_size=1),
            CMHSA(input_dim // 2)   # CMHSA expects (B, C//2, H, W), returns same
        )

        # Branch 2 => (B, C//2, H, W)
        self.branch2 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1)

        # Final processing after concatenation
        self.final_conv = CLB(input_dim)

    def forward(self, x):
        """
        x: (B, input_dim, H, W)
        """
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out_cat = torch.cat([out1, out2], dim=1)
        out = self.final_conv(out_cat)
        return out


class PermuteLayer(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)

class CMHSA(nn.Module):
    """
    A toy single-head convolutional self-attention mechanism.
    Expects input with `channels` and returns the same shape.
    """
    def __init__(self, channels: int):
        super().__init__()

        # For Q, K, V: we do a 1x1 conv to project from `channels` -> `channels`.
        # (In multi-head, you'd usually split channels, but let's keep it simple.)

        self.convQ = nn.Conv2d(channels, channels, kernel_size=1)
        self.convK = nn.Conv2d(channels, channels, kernel_size=1)
        self.convV = nn.Conv2d(channels, channels, kernel_size=1)

        # Final projection back to `channels`
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, channels, H, W)
        """
        B, C, H, W = x.shape

        # Compute Q, K, V
        Q = self.convQ(x)  # (B, C, H, W)
        K = self.convK(x)  # (B, C, H, W)
        V = self.convV(x)  # (B, C, H, W)

        # Flatten spatial dims so shapes become:
        #   Q => (B, C, H*W)
        #   K => (B, C, H*W)
        #   V => (B, C, H*W)
        Q = Q.flatten(start_dim=2)
        K = K.flatten(start_dim=2)
        V = V.flatten(start_dim=2)

        # We want (B, seq_len, C) for Q, or (B, C, seq_len) depending on how you do matmul.
        # Let's do (B, C, H*W) for K, and for Q => (B, H*W, C). So we permute Q and V:
        Q = Q.permute(0, 2, 1)  # (B, H*W, C)
        V = V.permute(0, 2, 1)  # (B, H*W, C)
        # Now shapes:
        #   Q => (B, H*W, C)
        #   K => (B, C, H*W)
        #   V => (B, H*W, C)

        # Compute attention scores: (B, H*W, H*W)
        scores = torch.matmul(Q, K)  # => (B, H*W, H*W)
        scores = scores / (C ** 0.5) # scale by sqrt(C)
        attn   = nn.Softmax(dim=-1)(scores)

        # Multiply attention map by V:
        # attn => (B, H*W, H*W)
        # V    => (B, H*W,  C)
        out = torch.matmul(attn, V)  # => (B, H*W, C)

        # Reshape back to (B, C, H, W)
        out = out.permute(0, 2, 1)          # => (B, C, H*W)
        out = out.view(B, C, H, W)         # => (B, C, H, W)

        # Final 1x1 conv if you like
        out = self.out_conv(out)           # => (B, C, H, W)

        return out





