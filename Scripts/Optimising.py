import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image
from medpy.metric.binary import hd
import matplotlib.pyplot as plt
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


# Define Dice coefficient function
def dice_score(pred, target):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)


# Define Hausdorff Distance function
def hausdorff_distance(pred, target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)

    # Check if both pred and target contain any binary object
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        # Return a large value or None to indicate an invalid distance
        return float('inf')  # Optionally, you could return `None` and handle it later

    return hd(pred_np, target_np)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=2).to(device)

# Criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Ensure the save directory exists
save_dir = '../Model weights'
os.makedirs(save_dir, exist_ok=True)
best_weights_path = os.path.join(save_dir, 'best_segmentation_model_weights_unetXresnet18.pth')

# Training and validation loop
num_epochs = 15

# Initialise variables
best_dice = -float('inf')
best_hausdorff = float('inf')

# Define a file path for saving metrics
csv_file_path = "../optimised_unetXresnet18.csv"
# Initialize the CSV file with headers
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Val Loss", "Dice Score", "Hausdorff Distance"])

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Training loop
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device).long().squeeze(1)
        optimizer.zero_grad()

        outputs = model(images)
        outputs = nn.functional.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}")

    # Validation loop
    model.eval()
    val_loss, dice_total, hausdorff_total = 0.0, 0.0, 0.0
    hausdorff_count = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device).long().squeeze(1)
            outputs = model(images)
            outputs = nn.functional.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
            val_loss += criterion(outputs, masks).item()

            preds = torch.argmax(outputs, dim=1)

            for pred, mask in zip(preds, masks):
                dice_val = dice_score(pred, mask).item()
                hausdorff_val = hausdorff_distance(pred, mask)

                dice_total += dice_val
                if hausdorff_val != float('inf'):
                    hausdorff_total += hausdorff_val
                    hausdorff_count += 1

    scheduler.step()
    avg_val_loss = val_loss / len(val_loader)
    avg_dice = float(dice_total / len(val_loader.dataset))
    avg_hausdorff = float(hausdorff_total / hausdorff_count) if hausdorff_count > 0 else float('inf')

    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss}, Dice: {avg_dice}, Hausdorff: {avg_hausdorff}")

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, avg_val_loss, avg_dice, avg_hausdorff])

    if avg_dice > best_dice or avg_hausdorff < best_hausdorff:
        best_dice = max(avg_dice, best_dice)
        best_hausdorff = min(avg_hausdorff, best_hausdorff)
        torch.save(model.state_dict(), best_weights_path)
        print(f"Best model saved with Dice: {avg_dice}, Hausdorff: {avg_hausdorff} at {best_weights_path}")

print(f"Metrics saved to {csv_file_path}")
print("--- Training complete: %s seconds ---" % (time.time() - start_time))

# Initialize lists to store the data
epochs = []
train_losses = []
val_losses = []

# Read the CSV file
csv_file_path = "../optimised_unetXresnet18.csv"
with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        epochs.append(int(row["Epoch"]))
        train_losses.append(float(row["Train Loss"]))
        val_losses.append(float(row["Val Loss"]))

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()