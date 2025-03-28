

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



class ObjectDetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=9):
        """
        in_channels: number of channels in the input feature map
        num_classes: number of target classes (excluding background if applicable)
        num_anchors: number of anchors per spatial location
        """
        super(ObjectDetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # In your model's __init__:
        # Classification branch: predicts class scores for each anchor
        self.cls_conv = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

        # Regression branch: predicts bounding box coordinates (e.g., [x, y, w, h]) for each anchor
        self.reg_conv = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        
        # Objectness branch: predicts a single score per anchor indicating object presence.
        self.obj_conv = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=3, padding=1)
        nn.init.constant_(self.obj_conv.bias, 2.0) 
    def forward(self, features):
        """
        features: a feature map tensor of shape (B, in_channels, H, W)
        Returns:
            cls_logits: tensor of shape (B, H * W * num_anchors, num_classes)
            bbox_reg: tensor of shape (B, H * W * num_anchors, 4)
        """
        cls_logits = self.cls_conv(features)  # Shape: (B, num_anchors*num_classes, H, W)
        bbox_reg = self.reg_conv(features)      # Shape: (B, num_anchors*4, H, W)
        obj_logits = self.obj_conv(features)    
        # Permute and reshape classification output to (B, H*W*num_anchors, num_classes)
        batch_size, _, H, W = cls_logits.shape
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(batch_size, -1, self.num_classes)

        # Permute and reshape regression output to (B, H*W*num_anchors, 4)
        bbox_reg = bbox_reg.permute(0, 2, 3, 1).contiguous()
        bbox_reg = bbox_reg.view(batch_size, -1, 4)
        # Reshape objectness output.
        
        
        obj_logits = obj_logits.permute(0, 2, 3, 1).contiguous()
        # Squeeze out the singleton channel dimension: final shape (B, H*W*num_anchors)
        obj_logits = obj_logits.view(batch_size, -1)


        return cls_logits, bbox_reg, obj_logits
