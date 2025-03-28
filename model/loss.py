
import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self, obj_loss_weight=1.0):
        super().__init__()
        # Classification loss: ignore anchors with label -1.
        self.loss_cls = nn.CrossEntropyLoss(ignore_index=-1)
        # Bounding box regression loss.
        self.loss_bbox = nn.SmoothL1Loss()
        # Objectness loss: using binary cross entropy with logits.
        self.loss_obj = nn.BCEWithLogitsLoss()
        self.obj_loss_weight = obj_loss_weight

    def forward(self, predictions, targets):
        # predictions are assumed to be a tuple of (cls_logits, bbox_reg, obj_logits)
        cls_logits, bbox_reg, obj_logits = predictions  
        # Expected shapes:
        #   cls_logits: [B, num_anchors, num_classes]
        #   bbox_reg:   [B, num_anchors, 4]
        #   obj_logits: [B, num_anchors]  (or [B, num_anchors, 1])
        B, num_anchors, num_classes = cls_logits.shape

        # Flatten classification predictions.
        cls_logits_flat = cls_logits.view(-1, num_classes)  # [B*num_anchors, num_classes]
        target_labels_flat = targets["labels"].view(-1)       # [B*num_anchors]
        loss_cls = self.loss_cls(cls_logits_flat, target_labels_flat)

        # Flatten bounding box predictions.
        bbox_reg_flat = bbox_reg.view(-1, 4)                  # [B*num_anchors, 4]
        target_bbox_flat = targets["bbox_targets"].view(-1, 4)  # [B*num_anchors, 4]
        loss_bbox = self.loss_bbox(bbox_reg_flat, target_bbox_flat)

        # For objectness, create a target where:
        #   1 indicates a positive anchor (object present),
        #   0 indicates a negative anchor (background).
        # We'll ignore anchors with label -1.
        mask = (target_labels_flat != -1)
        # Define objectness target: positive if label==1, negative otherwise.
        # (If you have more than two classes, adjust this logic as needed.)
        obj_target = (target_labels_flat == 1).float()
        # Flatten objectness predictions. Ensure obj_logits is a 1D tensor per anchor.
        obj_logits_flat = obj_logits.view(-1)
        loss_obj = self.loss_obj(obj_logits_flat[mask], obj_target[mask])

        total_loss = loss_cls + loss_bbox + self.obj_loss_weight * loss_obj
        return  total_loss ,loss_bbox ,loss_obj , loss_cls 