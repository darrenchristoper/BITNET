
import warnings

from sklearn.metrics import average_precision_score
from model.loss import DetectionLoss
from  model.model import BiTNet

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys
from model.utilsdetection import  decode_detections_multilabel_vectorized,match_anchors_to_gt
from model.preprocessdetection import VOCDetectionWithLabels, generate_and_normalize_anchors
from model.utils import plot_losses, plot_map5095
from torchmetrics.detection.mean_ap import MeanAveragePrecision
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)
def collate_targets(targets_list):
    # targets_list is a list of dictionaries, each with keys "labels" and "bbox_targets".
    batched_labels = []
    batched_bbox_targets = []
    batched_objectness=[]
    for target in targets_list:
        batched_labels.append(target["labels"])
        batched_bbox_targets.append(target["bbox_targets"])
        batched_objectness.append(target["objectness"])
    
    # This requires that all targets have the same shape. If not, you'll need to pad them.
    batched_labels = torch.stack(batched_labels, dim=0)
    batched_bbox_targets = torch.stack(batched_bbox_targets, dim=0)
    batched_objectness = torch.stack(batched_objectness, dim=0)
    
    return {"labels": batched_labels, "bbox_targets": batched_bbox_targets,  "objectness": batched_objectness}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print (sys.path)
    # Set device
    print(f"Using device: {device}")
    # Hyperparameters
    batch_size = 4
    num_workers = 2
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
 


    # Create datasets

    train_dataset = VOCDetectionWithLabels(
    root="./VOC2009data",
    year="2009",
    image_set="train",
    download=False,
)
    val_dataset = VOCDetectionWithLabels(
    root="./VOC2009data",
    year="2009",
    image_set="val",
    download=False,
)

    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn
)

    val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn
)
    # Get one batch from the train loader
    batch = next(iter(train_loader))
    images, targets = batch

 

    total_losses = []
    obj_losses = []
    local_losses = []
    class_losses = []
    map50_history=[]
    map5095_history=[]
    train_map_history = []
    val_map_history = []
    positive_counts = []
    # Initialize model
    model = BiTNet(num_classes=20)  # VOC has 20 classes + background
    model = model.to(device)
       # Initialize TorchMetrics for mAP@50 and mAP@50:95 on CUDA
    map50_metric = MeanAveragePrecision(iou_thresholds=[0.5]).to(device)
    map50_95_metric = MeanAveragePrecision().to(device)

# Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = DetectionLoss()
    # Define anchor parameters (adjust as needed)
    feature_map_size = (14, 14)   # Example feature map size
    scales = [32, 64, 128]  # Larger scales for 224x224 images
    ratios = [0.5, 1.0, 2.0]
    stride = 1
    image_size = (224, 224)
    

    anchors_np = generate_and_normalize_anchors(image_size, stride, scales, ratios)
    assert np.all(anchors_np <= 1.0), "Anchors not normalized!"
    # Set up model, optimizer, and loss
    for epoch in range(num_epochs):
        model.train()
        running_total_loss = 0.0
        running_bbox_loss = 0.0
        running_obj_loss = 0.0
        running_cls_loss = 0.0
        
        # Iterate over batches from the train_loader
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.to(device)
        
        # Forward pass: get predictions from the model
            cls_logits, bbox_reg,obj_logits = model(images)
        
            # For each image in the batch, match anchors to ground truth and compute offsets
            batch_detection_targets = []
            for target in targets:
            # Extract boxes and ensure they have shape [N, 4]
                boxes = target["boxes"]
                if boxes.dim() == 1:
                    boxes = boxes.unsqueeze(0)
                gt_boxes = boxes.numpy()  # convert to numpy for matching
                 # Ground truth boxes should be in [x_min, y_min, x_max, y_max] format

                assert np.all(gt_boxes <= 1.0), "Ground truth boxes not normalized!"
            
            # Extract labels and ensure they have shape [N]
                labels = target["labels"]
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                # Match anchors to ground truth boxes.
                # match_anchors_to_gt should return, for each anchor, a label (e.g., 1 for positive), the index of the matched ground-truth, etc.
                labels_np, matched_idx, _ = match_anchors_to_gt(anchors_np, gt_boxes)
           
                # Compute bbox_targets: for each anchor, if positive (labels_np[i]==1), compute the offset between gt_box and anchor.
                bbox_targets = np.zeros_like(anchors_np)
                positive_indices = np.where(labels_np == 1)[0]  # get indices of positive anchors
                if positive_indices.size > 0:
                    # For these indices, subtract anchor from corresponding gt box.
                    # gt_boxes[matched_idx[positive_indices]] gathers the matched gt box for each positive anchor.
                    bbox_targets[positive_indices] = gt_boxes[matched_idx[positive_indices]] - anchors_np[positive_indices]
            # Create an objectness target: binary (1 for positive, 0 for negative).
            # For anchors marked -1 (ignore), you can either set them to 0 or handle them later via masking.
                obj_targets = (labels_np == 1).astype(np.float32)
                positive_counts.append(np.sum(labels_np == 1))
               
                print("Objectness scores:", torch.sigmoid(obj_logits).mean().item())
               
            # Convert matching results back to tensors.
                target_dict = {
                    "labels": torch.tensor(labels_np, dtype=torch.long), 
                    "bbox_targets": torch.tensor(bbox_targets, dtype=torch.float32),
                    "objectness": torch.tensor(obj_targets, dtype=torch.float32)
                }
                batch_detection_targets.append(target_dict)
      
        
    
          
        # Compute loss using your detection loss function (criterion).
            predictions = (cls_logits, bbox_reg , obj_logits)
            batch_targets = collate_targets(batch_detection_targets)  # or however you collect targets
            print("Objectness targets mean:", batch_targets["objectness"].mean().item()) 
            batch_targets["labels"] = batch_targets["labels"].to(device)
            target_shape = batch_targets["labels"].view(-1).shape[0]
            pred_shape = cls_logits.view(-1, cls_logits.shape[-1]).shape[0]
            
            print(f"Predicted anchors: {pred_shape}, Target anchors: {target_shape}")
            batch_targets["bbox_targets"] = batch_targets["bbox_targets"].to(device)
            batch_targets["objectness"] = batch_targets["objectness"].to(device)
        
            total_loss ,loss_bbox ,loss_obj , loss_cls  = criterion(predictions, batch_targets)
        
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
      
            running_total_loss += total_loss.item()
            running_bbox_loss += loss_bbox.item()
            running_obj_loss += loss_obj.item()
            running_cls_loss += loss_cls.item()
   
    
        # Average losses over the number of batches.
        avg_total_loss = running_total_loss / len(train_loader)
        avg_bbox_loss = running_bbox_loss / len(train_loader)
        avg_obj_loss = running_obj_loss / len(train_loader)
        avg_cls_loss = running_cls_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Total Loss: {avg_total_loss :.4f} - Localization Loss: {avg_bbox_loss :.4f} - Objectness Loss: {avg_obj_loss :.4f} - Classification Loss: {avg_cls_loss :.4f}")
        total_losses.append(avg_total_loss )
        obj_losses.append(avg_obj_loss)
        class_losses.append(avg_cls_loss)
        local_losses.append(avg_bbox_loss)
        model.eval()
        preds = []         # to store predictions for each image
        targets_list = []  # to store ground truth for each image
    
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = images.to(device)
                cls_logits, bbox_reg, obj_logits = model(images)
        
             # Decode the detections for the batch
                batch_preds = decode_detections_multilabel_vectorized(cls_logits, bbox_reg, obj_logits, anchors_np, device=device)
                preds.extend(batch_preds)
        
        # Also, send ground truth targets to the same CUDA device if they're not already
                for target in targets:
                    if not target["boxes"].is_cuda:
                        target["boxes"] = target["boxes"].to(device)
                    if not target["labels"].is_cuda:
                        target["labels"] = target["labels"].to(device)
                targets_list.extend(targets)

 
    # Update the metrics with the predictions and ground truths
        map50_metric.update(preds, targets_list)
        map50_95_metric.update(preds, targets_list)
    
    # Compute the metrics
        results_map50 = map50_metric.compute()
        results_map50_95 = map50_95_metric.compute()
        epoch_map50 = results_map50["map"].item()  # convert to Python float if needed
        epoch_map50_95 = results_map50_95["map"].item()
        map50_history.append(epoch_map50)
        map5095_history.append(epoch_map50_95)
        print(f"Epoch {epoch+1} mAP@50: {epoch_map50:.4f}")
        print(f"Epoch {epoch+1} mAP@50:95: {epoch_map50_95:.4f}")
    plot_losses(total_losses=total_losses,local_losses=local_losses,obj_losses=obj_losses,class_losses=class_losses)
    plot_map5095(map_values50=map50_history,map_values5095=map5095_history)

    





def cuda():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Count:", torch.cuda.device_count())
        print("Current Device:", torch.cuda.current_device())
        print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))




if __name__ == "__main__":
    main()