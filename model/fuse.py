
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import ECAB

class BiFPN(nn.Module):
    """
    Minimal BiFPN-like module: merges four scales with a top-down pathway.
    We'll unify channels => out_channels, then do upsampling and summation.
    """
    def __init__(self, in_channels_list, out_channels, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon

        # Convs to unify channel dims to out_channels
        self.conv_p3 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.conv_p5 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.conv_p7 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)
        self.conv_p9 = nn.Conv2d(in_channels_list[3], out_channels, kernel_size=1)

        # Optional: "feature-wise" conv after merges
        self.smooth_p3 = ECAB(out_channels)
        self.smooth_p5 = ECAB(out_channels)
        self.smooth_p7 = ECAB(out_channels)
        self.smooth_p9 = ECAB(out_channels)

    def forward(self, inputs):
        """
        inputs: tuple of 4 feature maps (p3, p5, p7, p9),
        with shapes something like:
          p3: (B,  32, H   , W   )
          p5: (B,  64, H/2 , W/2 )
          p7: (B, 128, H/4 , W/4 )
          p9: (B, 256, H/8 , W/8 )

        We'll unify them to out_channels (e.g. 128) then do
        a single top-down pass:
            p9 -> upsample -> merges with p7
            p7 -> upsample -> merges with p5
            p5 -> upsample -> merges with p3
        Return (p3_out, p5_out, p7_out, p9_out).
        """
        p3_in, p5_in, p7_in, p9_in = inputs

        # 1) Unify channels
        p3 = self.conv_p3(p3_in)  # => (B, outC, H,   W)
        p5 = self.conv_p5(p5_in)  # => (B, outC, H/2, W/2)
        p7 = self.conv_p7(p7_in)  # => (B, outC, H/4, W/4)
        p9 = self.conv_p9(p9_in)  # => (B, outC, H/8, W/8)

        # 2) Top-down pathway
        # Upsample p9 to p7's size, then sum
        p9_up = F.interpolate(p9, size=p7.shape[2:], mode='nearest')
        p7_td = p7 + p9_up
        p7_relu = F.relu(p7_td)
        p7_td = self.smooth_p7(p7_relu)

        # Upsample p7_td to p5's size, then sum
        p7_up = F.interpolate(p7_td, size=p5.shape[2:], mode='nearest')
        p5_td = p5 + p7_up
        p5_relu = F.relu(p5_td)
        p5_td = self.smooth_p5(p5_relu)

        # Upsample p5_td to p3's size, then sum
        p5_up = F.interpolate(p5_td, size=p3.shape[2:], mode='nearest')
        p3_td = p3 + p5_up
        p3_relu = F.relu(p3_td)
        p3_td = self.smooth_p3(p3_relu)


        # 3) You might also want a smoothing conv on p9 itself
        p9_out = self.smooth_p9(p9)
        p7_out = p7_td
        p5_out = p5_td
        p3_out = p3_td

        return (p3_out, p5_out, p7_out, p9_out)
