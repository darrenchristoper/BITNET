
import numpy as np

import torch
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from tqdm import tqdm
import torch
import torchvision.ops as ops

def compute_iou(boxes1, boxes2):
    """
    Vectorized IoU calculation between anchor boxes (N,4) and GT boxes (M,4)
    Returns IoU matrix (N,M)
    """
    # Expand dimensions for broadcasting
    boxes1 = np.expand_dims(boxes1, 1)  # (N,1,4)
    boxes2 = np.expand_dims(boxes2, 0)  # (1,M,4)
    
    # Calculate intersection areas
    inter_xmin = np.maximum(boxes1[...,0], boxes2[...,0])
    inter_ymin = np.maximum(boxes1[...,1], boxes2[...,1])
    inter_xmax = np.minimum(boxes1[...,2], boxes2[...,2])
    inter_ymax = np.minimum(boxes1[...,3], boxes2[...,3])
    
    inter_w = np.maximum(inter_xmax - inter_xmin, 0)
    inter_h = np.maximum(inter_ymax - inter_ymin, 0)
    intersection = inter_w * inter_h

    # Calculate union areas
    area1 = (boxes1[...,2] - boxes1[...,0]) * (boxes1[...,3] - boxes1[...,1])
    area2 = (boxes2[...,2] - boxes2[...,0]) * (boxes2[...,3] - boxes2[...,1])
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    iou = np.where(union > 0, intersection / union, 0)
    return np.clip(iou, 0.0, 1.0)  # Ensure valid range


def match_anchors_to_gt(anchors, gt_boxes, pos_iou_thresh=0.5, neg_iou_thresh=0.4):
    iou_matrix = compute_iou(anchors, gt_boxes)
    print(iou_matrix)
    # 1. For each GT, find best matching anchor
    gt_best_anchor = iou_matrix.argmax(axis=0)
    
    # 2. For each anchor, find best matching GT
    anchor_best_gt = iou_matrix.argmax(axis=1)
    best_iou = iou_matrix.max(axis=1)
    
    # Initialize labels to ignore (-1)
    labels = np.full(anchors.shape[0], -1, dtype=np.int32)
    
    # 3. Mark positive anchors (two conditions)
    pos_mask = best_iou >= pos_iou_thresh
    pos_mask[gt_best_anchor] = True  # Force match for each GT
    
    # 4. Mark negative anchors
    neg_mask = best_iou < neg_iou_thresh
    labels[pos_mask] = 1
    labels[neg_mask] = 0
    
    return labels, anchor_best_gt, best_iou





def decode_detections_multilabel_vectorized(cls_logits, bbox_reg, obj_logits, anchors_np, 
                                              score_thresh=0.05, nms_thresh=0.5, max_detections=100, device=None):
    """
    Vectorized multi-label decoding.
    
    Args:
      cls_logits: Tensor of shape [batch_size, num_anchors, num_classes]
      bbox_reg: Tensor of shape [batch_size, num_anchors, 4]
      obj_logits: Tensor of shape [batch_size, num_anchors]
      anchors_np: NumPy array of shape [num_anchors, 4] containing anchor boxes.
      score_thresh: Minimum score threshold for a detection.
      nms_thresh: IoU threshold for non-maximum suppression.
      max_detections: Maximum detections per image.
      device: torch.device to use.
      
    Returns:
      List of dictionaries (one per image), each with:
         "boxes": Tensor of shape [N, 4]
         "scores": Tensor of shape [N]
         "labels": Tensor of shape [N]
    """
    batch_size = cls_logits.shape[0]
    # Create anchors tensor on device.
    anchors = torch.tensor(anchors_np, dtype=torch.float32, device=device)  # [num_anchors, 4]
    predictions = []
    
    for i in range(batch_size):
        # Vectorized computation over all anchors.
        cls_prob = torch.sigmoid(cls_logits[i])      # [num_anchors, num_classes]
        obj_prob = torch.sigmoid(obj_logits[i])        # [num_anchors]
        final_scores_all = cls_prob * obj_prob.unsqueeze(-1)  # [num_anchors, num_classes]
        pred_boxes = anchors + bbox_reg[i]             # [num_anchors, 4]
        
        final_boxes_list = []
        final_scores_list = []
        final_labels_list = []
        num_classes = cls_prob.shape[1]
        
        # Loop over classes (usually a small number).
        for c in range(num_classes):
            # For class c, get scores for all anchors.
            scores_c = final_scores_all[:, c]           # [num_anchors]
            mask = scores_c > score_thresh              # [num_anchors] boolean
            if mask.sum() == 0:
                continue  # Skip class c if no anchor exceeds threshold.
            
            boxes_c = pred_boxes[mask]                  # [num_valid, 4]
            scores_c = scores_c[mask]                   # [num_valid]
            
            # Apply NMS per class.
            keep = ops.nms(boxes_c, scores_c, nms_thresh)
            boxes_c = boxes_c[keep]
            scores_c = scores_c[keep]
            
    
            final_boxes_list.append(boxes_c)
            final_scores_list.append(scores_c)
            final_labels_list.append(torch.full((len(keep),), c, device=device, dtype=torch.long))
         
        if final_boxes_list:
            final_boxes = torch.cat(final_boxes_list, dim=0)
            final_scores = torch.cat(final_scores_list, dim=0)
            final_labels = torch.cat(final_labels_list, dim=0)
            
            # Sort detections by score (descending) and limit to max_detections.
            if final_boxes.shape[0] > max_detections:
                final_scores, idx = final_scores.sort(descending=True)
                idx = idx[:max_detections]
                final_boxes = final_boxes[idx]
                final_scores = final_scores[idx]
                final_labels = final_labels[idx]
        else:
            final_boxes = torch.empty((0, 4), device=device)
            final_scores = torch.empty((0,), device=device)
            final_labels = torch.empty((0,), device=device, dtype=torch.long)
       
        predictions.append({
            "boxes": final_boxes,
            "scores": final_scores,
            "labels": final_labels
        })
     
    return predictions
