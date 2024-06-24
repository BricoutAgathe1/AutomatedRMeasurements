import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import time
import json
from sklearn.model_selection import ParameterGrid, KFold

start_time = time.time()

# Initialize hyperparameter grid
param_grid = {
    'learning_rate': [0.001, 0.0001],
    'batch_size': [8, 16],
    'num_epochs': [20, 30],
}

param_combinations = list(ParameterGrid(param_grid))
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Dataset class
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
        mask_name = img_name.replace('.jpg', '_mask.png')
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
train_image_dir = 'data/train'
train_mask_dir = 'data/train_masks'
val_image_dir = 'data/val'
val_mask_dir = 'data/val_masks'
unlabeled_train_image_dir = 'data/train_unlabeled'

# Initialize datasets
train_dataset = UltrasoundDataset(train_image_dir, train_mask_dir, image_transforms)
val_dataset = UltrasoundDataset(val_image_dir, val_mask_dir, image_transforms)
unlabeled_train_dataset = UltrasoundDataset(unlabeled_train_image_dir, train_mask_dir, image_transforms)


# Create DataLoaders
def get_data_loaders(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Visualization to verify data loading
def visualize_dataset(dataset, num_samples=5):
    for i in range(num_samples):
        image, mask = dataset[i]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0))  # Permute the image to (H, W, C)
        plt.title("Image")

        plt.subplot(1, 2, 2)
        plt.imshow(mask.squeeze(), cmap='gray')  # Directly visualize the mask
        plt.title("Mask")

        plt.show()


visualize_dataset(train_dataset)


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


# Training loop with hyperparameter tuning
def train_model(params, train_loader, val_loader, device):
    model = ResNetSegmentation(backbone='resnet34', num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    best_val_loss = float('inf')
    for epoch in range(params['num_epochs']):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)  # Ensure masks are of type long and remove extra channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            outputs = nn.functional.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).long().squeeze(1)
                outputs = model(images)
                outputs = nn.functional.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_weights_hypertuning.pth')

    return best_val_loss


# Hyperparameter tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_params = None
best_val_loss = float('inf')

train_indices = list(range(len(train_dataset)))

for params in param_combinations:
    avg_val_loss = 0.0

    for train_index, val_index in kf.split(train_indices):
        train_subset = torch.utils.data.Subset(train_dataset, train_index)
        val_subset = torch.utils.data.Subset(train_dataset, val_index)

        train_loader = get_data_loaders(train_subset, params['batch_size'], shuffle=True)
        val_loader = get_data_loaders(val_subset, params['batch_size'], shuffle=False)

        val_loss = train_model(params, train_loader, val_loader, device)
        avg_val_loss += val_loss

    avg_val_loss /= kf.get_n_splits()
    print(f"Params: {params}, Avg Val Loss: {avg_val_loss}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_params = params

print(f"Best Params: {best_params}, Best Val Loss: {best_val_loss}")
