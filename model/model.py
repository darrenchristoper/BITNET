
from .model_backbone import BITNETBackbone
from .fuse import BiFPN
from .detect import DetectionHead
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiTNet(nn.Module):
    def __init__(self, num_classes=21):  # Changed to 21 for VOC (20 classes + background)
        super(BiTNet, self).__init__()
        # 1) Backbone
        self.backbone = BITNETBackbone(in_channels=3, base_channels=32)

        # 2) BiFPN
        self.bifpn = BiFPN(in_channels_list=[32, 64, 128, 256], out_channels=128)

        # 3) Detection Head - Remove the num_anchors parameter
        self.head = DetectionHead(in_channels=128, num_classes=num_classes)

    def forward(self, x):
        # Extract multi-scale features from backbone
        feat3, feat5, feat7, feat9 = self.backbone(x)

        # Pass features to BiFPN
        bifpn_inputs = (feat3, feat5, feat7, feat9)
        fused_feats = self.bifpn(bifpn_inputs)

        # Modified to match your DetectionHead implementation
        segmentation_output = self.head(fused_feats)

        return segmentation_output
    


