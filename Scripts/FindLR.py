import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image
from medpy.metric.binary import hd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import csv
import time

start_time = time.time()


# Dataset class
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
            potential_mask_path = os.path.join(mask_dir, img_name.replace('.jpg', '_mask.png'))
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
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x+0.05 * torch.randn_like(x)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 2.0)),
])


# Create dataloaders
batch_size = 8

train_image_dirs = ['../Datasets/Testing/Global_Dataset/train/half_flipped']
train_mask_dirs = ['../Datasets/Testing/Global_Dataset/train/half_flipped']

train_dataset = UltrasoundDataset(train_image_dirs, train_mask_dirs, image_transform=transform, mask_transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_image_dirs = ['../Datasets/Testing/Global_Dataset/val/half_flipped']
val_mask_dirs = ['../Datasets/Testing/Global_Dataset/val/half_flipped']

val_dataset = UltrasoundDataset(val_image_dirs, val_mask_dirs, image_transform=transform, mask_transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


def find_lr(model, train_loader, criterion, min_lr=1e-6, max_lr=1, num_iter=100):
    """
    Implements the learning rate finder.
    """
    optimizer = optim.SGD(model.parameters(), lr=min_lr)

    # Prepare variables to store loss values and learning rates
    lrs = []
    losses = []

    # Set model to training mode
    model.train()

    # Initially set the learning rate
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min_lr + (
                max_lr - min_lr) * epoch / num_iter)

    for i, (inputs, labels) in enumerate(train_loader):
        if i >= num_iter:
            break

        inputs, labels = inputs.cuda(), labels.cuda()
        labels = labels.squeeze(1)
        labels = labels.long()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()

        # Record the learning rate and loss
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        # Update the learning rate
        lr_scheduler.step()

    return lrs, losses

def plot_lr_find(lrs, losses):
    # Convert the lists to numpy arrays
    lrs = np.array(lrs)
    losses = np.array(losses)

    # Plot the loss as a function of the learning rate
    plt.figure(figsize=(8, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')  # Use a logarithmic scale for the x-axis
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')

    # Optionally, add a vertical line at the optimal learning rate (where the loss starts decreasing fastest)
    plt.axvline(x=lrs[np.argmin(losses)], color='r', linestyle='--', label="Optimal LR")
    plt.legend()

    plt.show()

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=2).to(device)

# Load previous weights if resuming training for fine-tuning
previous_weights_path = '../Model weights/best_segmentation_model_weights_unetXresnet18.pth'
if os.path.exists(previous_weights_path):
    model.load_state_dict(torch.load(previous_weights_path))

# Criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()

# Assuming train_loader is defined somewhere
lrs, losses = find_lr(model, train_loader, criterion)

# Plot the learning rate vs. loss curve
plot_lr_find(lrs, losses)