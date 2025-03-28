import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
# Define the VOC classes and mapping to indices.
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
)
VOC_CLASSES_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

class VOCDetectionWithLabels(torch.utils.data.Dataset):
    def __init__(self, root, year, image_set, download, TARGET_IMAGE_SIZE=(224, 224)):
        self.voc = torchvision.datasets.VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
        )
        self.TARGET_IMAGE_SIZE = TARGET_IMAGE_SIZE
    
    def __getitem__(self, idx):
        image, target = self.voc[idx]
        
        # Extract objects from the annotation.
        objects = target["annotation"].get("object")
        boxes = []
        labels = []
        if objects is not None:
            # In case a single object is present, wrap it in a list.
            if not isinstance(objects, list):
                objects = [objects]
            for obj in objects:
                bndbox = obj["bndbox"]
                # Convert bounding box coordinates to integers.
                xmin = int(bndbox["xmin"])
                ymin = int(bndbox["ymin"])
                xmax = int(bndbox["xmax"])
                ymax = int(bndbox["ymax"])
                boxes.append([xmin, ymin, xmax, ymax])
                # Convert class name to index.
                label_name = obj["name"]
                labels.append(VOC_CLASSES_TO_IDX[label_name])
        
        # Convert lists to tensors.
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Convert image from PIL to tensor and resize.
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.resize(image_tensor, self.TARGET_IMAGE_SIZE)
        
        # Normalize bounding box coordinates to [0, 1] using original image dimensions.
        # image.size returns (width, height)
        orig_w, orig_h = image.size  
        # Normalize x coordinates (xmin, xmax) by width and y coordinates by height.
        boxes[:, [0, 2]] /= orig_w  
        boxes[:, [1, 3]] /= orig_h
        
        target_dict = {"boxes": boxes, "labels": labels}
        return image_tensor, target_dict
    
    def __len__(self):
        return len(self.voc)
    
    


def generate_anchors(image_size, stride, scales, ratios):
    """
    Generate anchors for an image of size (height, width) with a given stride,
    using specified scales and aspect ratios.
    
    Args:
        image_size (tuple): (height, width) of the image.
        stride (int): Stride with which to slide over the image.
        scales (list of float): List of anchor scales (e.g., [32, 64, 128]).
        ratios (list of float): List of aspect ratios (e.g., [0.5, 1.0, 2.0]).
        
    Returns:
        anchors (np.ndarray): Array of shape (N, 4) in [x_min, y_min, x_max, y_max] format.
    """
    height, width = image_size
    fm_height = height // stride  # number of positions along height
    fm_width  = width // stride   # number of positions along width
    anchors = []
    
    for i in range(fm_height):
        for j in range(fm_width):
            # Center of the anchor in pixel coordinates.
            center_y = (i + 0.5) * stride
            center_x = (j + 0.5) * stride
            for scale in scales:
                for ratio in ratios:
                    w_anchor = scale * np.sqrt(ratio)
                    h_anchor = scale / np.sqrt(ratio)
                    x_min = center_x - w_anchor / 2
                    y_min = center_y - h_anchor / 2
                    x_max = center_x + w_anchor / 2
                    y_max = center_y + h_anchor / 2
                    anchors.append([x_min, y_min, x_max, y_max])
    anchors = np.array(anchors, dtype=np.float32)
    return anchors

def generate_and_normalize_anchors(image_size, stride, scales, ratios):
    anchors = generate_anchors(image_size, stride, scales, ratios)
    
    # Clip to image boundaries
    anchors[:, 0] = np.clip(anchors[:, 0], 0, image_size[1])  # x_min
    anchors[:, 1] = np.clip(anchors[:, 1], 0, image_size[0])    # y_min
    anchors[:, 2] = np.clip(anchors[:, 2], 0, image_size[1])    # x_max
    anchors[:, 3] = np.clip(anchors[:, 3], 0, image_size[0])    # y_max
    
    # Normalize
    anchors[:, [0, 2]] /= image_size[1]  # x coordinates
    anchors[:, [1, 3]] /= image_size[0]  # y coordinates
    return anchors