

import torch
import torch.nn as nn
import torch.nn.functional as F



class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features, target_size=(224, 224)):
        x = features[0]
        x = self.conv(x)
        x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return x

# And in your BiTNet forward method:
def forward(self, x):
    # Get input size for proper upsampling
    _, _, H, W = x.shape

    # Extract multi-scale features from backbone
    feat3, feat5, feat7, feat9 = self.backbone(x)

    # Pass features to BiFPN
    bifpn_inputs = (feat3, feat5, feat7, feat9)
    fused_feats = self.bifpn(bifpn_inputs)

    # Pass target size to head
    segmentation_output = self.head(fused_feats, target_size=(H, W))

    return segmentation_output