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
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '_mask.png')  # Ensure correct mask path
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0).float()  # Ensure binary values 0 and 1

        return image, mask


# Define image and mask transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

mask_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Paths to datasets
# train_image_dir = 'data/train'
# train_mask_dir = 'data/train_masks'
# val_image_dir = 'data/val'
# val_mask_dir = 'data/val_masks'
# unlabeled_train_image_dir = 'data/train_unlabeled'

# Paths to cropped datasets
train_image_dir = '../Datasets/Testing/Global_Dataset/train/cropped_images'
train_mask_dir = '../Datasets/Testing/Global_Dataset/train/cropped_images'
val_image_dir = '../Datasets/Testing/Global_Dataset/val/cropped_images'
val_mask_dir = '../Datasets/Testing/Global_Dataset/val/cropped_images'
unlabeled_train_image_dir = '../Datasets/Testing/Global_Dataset/ToPseudolabel'

# Initialize datasets
train_dataset = UltrasoundDataset(train_image_dir, train_mask_dir, image_transforms)
val_dataset = UltrasoundDataset(val_image_dir, val_mask_dir, image_transforms)
unlabeled_train_dataset = UltrasoundDataset(unlabeled_train_image_dir, train_mask_dir, image_transforms)

# Create DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
unlabeled_dataloader = DataLoader(unlabeled_train_dataset, batch_size=batch_size, shuffle=False)


# Define the ResNet-based segmentation model
class ResNetSegmentation(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=2):
        super(ResNetSegmentation, self).__init__()
        if backbone == 'resnet18':
            self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            encoder_output_dim = 512
        elif backbone == 'resnet34':
            self.encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            encoder_output_dim = 512
        elif backbone == 'resnet50':
            self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            encoder_output_dim = 2048
        elif backbone == 'resnet101':
            self.encoder = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            encoder_output_dim = 2048
        elif backbone == 'resnet152':
            self.encoder = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            encoder_output_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  # Remove the fully connected layer

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
model = ResNetSegmentation(backbone='resnet18', num_classes=2)  # Choose backbone as needed
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Pseudolabel the unlabeled training data
model.load_state_dict(torch.load('../Model weights/best_segmentation_model_weights.pth'))
model.eval()  # Set the model to evaluation mode


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

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name


# transform for pseudolabeling
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# unlabeled_image_dir = 'data/train_unlabeled'
unlabeled_image_dir = '../Datasets/Testing/Global_Dataset/ToPseudolabel'
unlabeled_dataset = UnlabeledUltrasoundDataset(unlabeled_image_dir, transform)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=False)

# pseudolabel_dir = 'data/train_pseudolabels'
pseudolabel_dir = '../Datasets/Testing/Global_Dataset/Pseudolabeled'
os.makedirs(pseudolabel_dir, exist_ok=True)


def extract_top_bottom_positions(mask):
    # Convert to grayscale if needed
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Find non-zero rows where mask is present
    non_zero_rows = np.where(mask > 0)[0]

    if len(non_zero_rows) > 0:
        # First non-zero row
        top_position = non_zero_rows[0]
        # Last non-zero row
        bottom_position = non_zero_rows[-1]
        return top_position, bottom_position
    else:
        return None, None  # No mask found


distances = []

with torch.no_grad():
    for images, image_names in unlabeled_dataloader:
        images = images.to(device)
        outputs = model(images)
        # get predicted class for each pixel
        pseudolabels = torch.argmax(outputs, dim=1)

        for pseudolabel, image_name in zip(pseudolabels, image_names):
            pseudolabel_np = pseudolabel.cpu().numpy().astype(np.uint8)
            pseudolabel_img = Image.fromarray(pseudolabel_np * 255)  # Scale to 255
            # pseudolabel_img = pseudolabel_img.resize((1280, 720), Image.NEAREST) # uncomment if not using cropped data

            if np.any(pseudolabel_np):
                pseudolabel_path = os.path.join(pseudolabel_dir, image_name.replace('.jpg', '_pseudolabel.png'))
                pseudolabel_img.save(pseudolabel_path)
                print(f"Pseudolabel saved: {pseudolabel_path}")

                top_pos, bottom_pos = extract_top_bottom_positions(pseudolabel_np)
                if top_pos is not None and bottom_pos is not None:
                    image_height = pseudolabel_np.shape[0]
                    distance_from_top = top_pos
                    distance_from_bottom = bottom_pos

                    distances.append({
                        'MaskFile': pseudolabel_path,
                        'DistanceFromTop': distance_from_top,
                        'DistanceFromBottom': distance_from_bottom
                    })
                else:
                    print(f"No mask found or mask is empty for {pseudolabel_path}")
            else:
                print(f"Warning: Empty pseudolabel for image {image_name}")
                pseudolabel_path = os.path.join(pseudolabel_dir, image_name.replace('.jpg', '_pseudolabel.png'))
                pseudolabel_img.save(pseudolabel_path)
                distances.append({
                    'MaskFile': pseudolabel_path,
                    'DistanceFromTop': -1,  # Indicate no mask found
                    'DistanceFromBottom': -1  # Indicate no mask found
                })


print("--- Pseudolabeling complete: %s seconds ---" % (time.time() - start_time))