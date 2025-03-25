
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import ECAB,ETB
class BITNETBackbone(nn.Module):

        def __init__(self,  in_channels=3, base_channels=32):
            super(BITNETBackbone, self).__init__()
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
            self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
            self.conv3 = ECAB(base_channels)
            self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
            self.conv5 = ECAB(base_channels*2)
            # Provide kernel_size, in_channels, out_channels, and kernel_size for conv6 layers
            self.conv6 = nn.Sequential (nn.MaxPool2d(kernel_size=2, stride=2), # Added kernel_size and stride
                               nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1)) # Added in_channels, out_channels, kernel_size, padding
            self.conv7 = ETB(base_channels*4)
            self.conv8 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True)
        )
            self.conv9 = ETB(base_channels*8)


        def forward(self, inputs): # Changed foward to forward
            out1 = self.conv1(inputs)
            out2  = self.conv2(out1)
            out3 = self.conv3(out2)
            out4 = self.conv4 (out3)
            out5 = self.conv5 (out4) # Changed out5 to out4, to take output from previous layer
            out6 = self.conv6 (out5)
            out7 = self.conv7(out6) # Changed self.conv to self.conv7
            out8 = self.conv8(out7) # Changed self.conv to self.conv8
            out9 = self.conv9(out8) # Changed self.conv to self.conv9
                           # p1,p2,p3,p4

            return out3, out5, out7, out9