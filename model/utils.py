import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score
import torch
from  torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
def plot_map(map_values, val_map_values=None, test_map=None):
    """
    Plot mean Average Precision (mAP) over epochs.

    Parameters:
    map_values (list): List of mAP values for training data across epochs
    val_map_values (list, optional): List of mAP values for validation data across epochs
    test_map (float, optional): Final test mAP value to display as a horizontal line
    """
    plt.figure(figsize=(10, 6))

    # Plot training mAP
    epochs = range(1, len(map_values) + 1)
    plt.plot(epochs, map_values, 'b-', label='Training mAP', marker='o')

    # Plot validation mAP if provided
    if val_map_values:
        plt.plot(epochs, val_map_values, 'r-', label='Validation mAP', marker='x')

    # Add horizontal line for test mAP if provided
    if test_map is not None:
        plt.axhline(y=test_map, color='g', linestyle='--', label=f'Test mAP: {test_map:.4f}')

    plt.title('Mean Average Precision (mAP) over Training')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.ylim([0, 1.0])  # mAP is typically between 0 and 1
    plt.grid(True)
    plt.legend()

    # Add data points values as text
    for i, j in enumerate(map_values):
        plt.annotate(f'{j:.3f}', (epochs[i], j), textcoords="offset points",
                    xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.show()

def plot_map50(map_values):
    """
    Plot mean Average Precision (mAP) over epochs.

    Parameters:
    map_values (list): List of mAP values for training data across epochs
    val_map_values (list, optional): List of mAP values for validation data across epochs
    test_map (float, optional): Final test mAP value to display as a horizontal line
    """
    plt.figure(figsize=(10, 6))

    # Plot training mAP
    epochs = range(1, len(map_values) + 1)
    plt.plot(epochs, map_values, 'b-', label='mAP@50', marker='o')

   

    plt.title('Mean Average Precision (mAP) over Training ')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.ylim([0, 1.0])  # mAP is typically between 0 and 1
    plt.grid(True)
    plt.legend()

    # Add data points values as text
    for i, j in enumerate(map_values):
        plt.annotate(f'{j:.3f}', (epochs[i], j), textcoords="offset points",
                    xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.show()

def plot_map5095(map_values):
    """
    Plot mean Average Precision (mAP) over epochs.

    Parameters:
    map_values (list): List of mAP values for training data across epochs
    val_map_values (list, optional): List of mAP values for validation data across epochs
    test_map (float, optional): Final test mAP value to display as a horizontal line
    """
    plt.figure(figsize=(10, 6))

    # Plot training mAP
    epochs = range(1, len(map_values) + 1)
    plt.plot(epochs, map_values, 'b-', label='mAP@50:95 ', marker='o')

    

    plt.title('Mean Average Precision (mAP) over Training ')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.ylim([0, 1.0])  # mAP is typically between 0 and 1
    plt.grid(True)
    plt.legend()

    # Add data points values as text
    for i, j in enumerate(map_values):
        plt.annotate(f'{j:.3f}', (epochs[i], j), textcoords="offset points",
                    xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, val_losses=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
# Function to evaluate model on test dataset
def evaluate_model(model, test_loader, device, criterion=None, train_losses=None, val_losses=None):
    model.eval()
    test_loss = 0.0
    all_ious = []

    color_map = create_pascal_label_colormap()
    # Store a single example for visualization
    example_for_viz = None

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.squeeze(1).long()  # (B, H, W)

            # Forward pass
            outputs = model(images)  # (B, C, H, W)

            # Calculate loss if criterion provided
            if criterion:
                loss = criterion(outputs, masks)
                test_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs, 1)  # (B, H, W)

            # Calculate IoU for each image in batch
            for j in range(images.size(0)):
                iou = calculate_iou(preds[j].cpu().numpy(), masks[j].cpu().numpy())
                all_ious.append(iou)

            # Store just one example for visualization later
            if example_for_viz is None:
                example_for_viz = {
                    'image': images[0].cpu(),
                    'mask': masks[0].cpu().numpy(),
                    'pred': preds[0].cpu().numpy()
                }

    # Calculate mean IoU
    mean_iou = np.mean(all_ious)

    # Calculate average loss if criterion provided
    avg_loss = test_loss / len(test_loader) if criterion else None

    # Now display the visualization and loss plot at the end
    plt.figure(figsize=(15, 10))

    # Plot loss if provided
    if train_losses is not None:
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        if val_losses is not None:
            plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

    # Plot segmentation visualization if we have an example
    if example_for_viz is not None:
        if train_losses is not None:
            plt.subplot(2, 1, 2)

        # Convert tensor to numpy for visualization
        image = example_for_viz['image'].permute(1, 2, 0).numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Apply colormap to mask and prediction
        mask = example_for_viz['mask']
        pred = example_for_viz['pred']
        colored_mask = color_map[mask]
        colored_pred = color_map[pred]

        # Create a horizontal layout for the segmentation visualization
        plt.subplot(2, 3, 4)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(colored_mask)
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(colored_pred)
        plt.title('Prediction')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    return avg_loss, mean_iou

# Function to calculate IoU for segmentation
def calculate_iou(pred, target, n_classes=21):
    ious = []
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()

        if union == 0:
            ious.append(float('nan'))  # Class not present
        else:
            ious.append(float(intersection) / float(union))

    # Return mean IoU over classes that are present
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0

# Function to create color map for visualization
def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark."""
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in range(8):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << (7-shift)
        ind >>= 3

    return colormap

# Function to visualize segmentation results
def visualize_prediction(image, mask, pred, colormap):
    plt.figure(figsize=(15, 5))

    # Convert tensor to numpy for visualization
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # CHW -> HWC
        # Normalize image for display if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

    # Apply colormap to mask and prediction
    colored_mask = colormap[mask]
    colored_pred = colormap[pred]

    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')

    # Plot ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(colored_mask)
    plt.title('Ground Truth')
    plt.axis('off')

    # Plot prediction
    plt.subplot(1, 3, 3)
    plt.imshow(colored_pred)
    plt.title('Prediction')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def calculate_map_segmentation_per_batch(model, data_loader, device, num_classes, return_per_class=False, verbose=True):
    """
    Calculate mean Average Precision (mAP) for semantic segmentation incrementally per batch.

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to evaluate
    data_loader : torch.utils.data.DataLoader
        DataLoader containing the evaluation dataset
    device : torch.device
        Device to run the model on (CPU or GPU)
    num_classes : int
        Number of classes in the dataset
    return_per_class : bool, default=False
        If True, returns AP for each class along with   the mean
    verbose : bool, default=True
        If True, shows a progress bar during evaluation

    Returns:
    --------
    float or tuple
        If return_per_class is False, returns mean AP across all classes
        If return_per_class is True, returns (mean_ap, ap_per_class)
    """
    model.eval()

    # Initialize accumulators for AP statistics
    class_sums = [0.0 for _ in range(num_classes)]
    class_counts = [0 for _ in range(num_classes)]

    data_iter = tqdm(data_loader, desc="Calculating mAP", disable=not verbose) if verbose else data_loader

    with torch.no_grad():
        for images, masks in data_iter:
            images = images.to(device)
            masks = masks.to(device).long()

            # Forward pass
            outputs = model(images)
            predictions = torch.softmax(outputs, dim=1)  # (B, C, H, W)

            # Reshape to pixel-wise samples
            B, C, H, W = predictions.shape
            predictions_flat = predictions.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            masks_flat = masks.view(-1)  # (B*H*W,)

            # Convert masks to one-hot encoding
            masks_one_hot = torch.zeros(B*H*W, num_classes, device=device)
            masks_one_hot.scatter_(1, masks_flat.unsqueeze(1), 1)

            # Calculate AP for each class in this batch
            batch_ap = [0.0] * num_classes
            valid_in_batch = [False] * num_classes

            for c in range(num_classes):
                y_true = masks_one_hot[:, c].cpu().numpy()
                y_pred = predictions_flat[:, c].cpu().numpy()

                # Check if class is present in this batch
                if y_true.sum() > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            ap = average_precision_score(y_true, y_pred)
                            batch_ap[c] = ap
                            valid_in_batch[c] = True
                        except:
                            batch_ap[c] = 0.0
                else:
                    batch_ap[c] = 0.0

            # Update running totals
            for c in range(num_classes):
                if valid_in_batch[c]:
                    class_sums[c] += batch_ap[c]
                    class_counts[c] += 1

    # Calculate final AP for each class
    ap_per_class = []
    valid_classes = 0
    total_ap = 0.0

    for c in range(num_classes):
        if class_counts[c] > 0:
            ap = class_sums[c] / class_counts[c]
        else:
            ap = 0.0

        ap_per_class.append(ap)

        if class_counts[c] > 0:
            valid_classes += 1
            total_ap += ap

    # Calculate mean AP
    mean_ap = total_ap / valid_classes if valid_classes > 0 else 0.0

    if return_per_class:
        return mean_ap, ap_per_class
    else:
        return mean_ap
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import warnings
import numpy as np

def calculate_map_iou_thresholds_per_batch(model, data_loader, device, num_classes, return_per_class=False, verbose=True):
    model.eval()

    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.50, 0.55, ..., 0.95]
    class_ap_sums = [[0.0 for _ in iou_thresholds] for _ in range(num_classes)]
    class_counts = [0 for _ in range(num_classes)]

    data_iter = tqdm(data_loader, desc="Calculating mAP@IoU", disable=not verbose) if verbose else data_loader

    with torch.no_grad():
        for images, masks in data_iter:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)  # (B, C, H, W)

            B, C, H, W = probs.shape
            preds = torch.argmax(probs, dim=1)  # (B, H, W)

            for b in range(B):
                pred_mask = preds[b]
                true_mask = masks[b]

                for cls in range(num_classes):
                    pred_binary = (pred_mask == cls)
                    true_binary = (true_mask == cls)

                    if true_binary.sum() == 0:
                        continue  # Skip if class not in ground truth

                    intersection = (pred_binary & true_binary).sum().item()
                    union = (pred_binary | true_binary).sum().item()
                    iou = intersection / union if union != 0 else 0.0

                    for i, threshold in enumerate(iou_thresholds):
                        ap = 1.0 if iou >= threshold else 0.0
                        class_ap_sums[cls][i] += ap
                    class_counts[cls] += 1

    ap_per_class = []
    map_50_total = 0.0
    map_5095_total = 0.0
    valid_classes = 0

    for cls in range(num_classes):
        if class_counts[cls] > 0:
            aps = [ap_sum / class_counts[cls] for ap_sum in class_ap_sums[cls]]
            ap_per_class.append(aps)

            map_50_total += aps[0]  # IoU=0.50
            map_5095_total += sum(aps) / len(iou_thresholds)
            valid_classes += 1
        else:
            ap_per_class.append([0.0 for _ in iou_thresholds])

    map_50 = map_50_total / valid_classes if valid_classes > 0 else 0.0
    map_5095 = map_5095_total / valid_classes if valid_classes > 0 else 0.0

    if return_per_class:
        return map_50, map_5095, ap_per_class  # ap_per_class: List[List[AP@each_threshold]]
    else:
        return map_50, map_5095
