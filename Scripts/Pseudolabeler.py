import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageTk
import matplotlib.pyplot as plt
import time
import json
from tkinter import Tk, Label, Canvas, Entry, Button

start_time = time.time()


class UltrasoundDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, image_transform=None, mask_transform=None):
        self.image_dirs = image_dirs  # List of image directories
        self.mask_dirs = mask_dirs  # List of mask directories
        self.image_filenames = []

        # Collect all filenames from each directory
        for image_dir in self.image_dirs:
            self.image_filenames.extend([os.path.join(image_dir, f) for f in os.listdir(image_dir)])

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        img_name = os.path.basename(img_path)

        # Identify the correct mask path by finding which mask directory contains the corresponding mask
        mask_path = None
        for mask_dir in self.mask_dirs:
            potential_mask_path = os.path.join(mask_dir, img_name.replace('.bmp', '_mask.png'))
            if os.path.exists(potential_mask_path):
                mask_path = potential_mask_path
                break

        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image: {img_path}")

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

            mask = (mask > 0).float()  # Binarize the mask

        return image, mask


# Define image and mask transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Define the ResNet-based segmentation model
class ResNetSegmentation(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=2, in_channels=1):
        super(ResNetSegmentation, self).__init__()

        # Choose ResNet backbone dynamically
        backbone_dict = {
            'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
            'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
            'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 2048),
            'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1, 2048),
            'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1, 2048)
        }

        if backbone not in backbone_dict:
            raise ValueError(f"Unsupported backbone: {backbone}")

        model_fn, weights, encoder_output_dim = backbone_dict[backbone]
        self.encoder = model_fn(weights=weights)

        # Modify first conv layer to handle grayscale input if necessary
        if in_channels == 1:
            old_conv1 = self.encoder.conv1
            self.encoder.conv1 = nn.Conv2d(
                in_channels, old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=False
            )
            # Copy pretrained weights (average across RGB channels)
            with torch.no_grad():
                self.encoder.conv1.weight = nn.Parameter(old_conv1.weight.sum(dim=1, keepdim=True))

        # Remove FC layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # Simple decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_output_dim, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Example usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetSegmentation(backbone='resnet152', num_classes=2).to(device)
model.load_state_dict(torch.load('../Model weights/MUIA_model_weights_resnet152.pth', map_location=torch.device('cpu')))
model.eval()



class UnlabeledUltrasoundDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, img_name


# Pseudolabelling settings
unlabeled_image_dir = '../Datasets/TM_Split/test/cropped_all'
pseudolabel_dir = '../Datasets/TM_Split/test/croppedAll_resnet152'
os.makedirs(pseudolabel_dir, exist_ok=True)

# Pseudolabelling process
unlabeled_dataset = UnlabeledUltrasoundDataset(unlabeled_image_dir, transform)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=False)


with torch.no_grad():
    for images, image_names in unlabeled_dataloader:
        images = images.to(device)
        outputs = model(images)
        pseudolabels = torch.argmax(outputs, dim=1)

        for pseudolabel, image_name in zip(pseudolabels, image_names):
            pseudolabel_np = (pseudolabel.cpu().numpy() * 255).astype(np.uint8)
            pseudolabel_path = os.path.join(pseudolabel_dir, image_name.replace('.bmp', '_pseudolabel.png'))
            cv2.imwrite(pseudolabel_path, pseudolabel_np)


print("--- Pseudolabeling complete: %s seconds ---" % (time.time() - start_time))