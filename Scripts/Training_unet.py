import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from medpy.metric.binary import hd
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

        image = Image.open(img_path).convert("L")  # Convert image to grayscale
        mask = Image.open(mask_path).convert("L")  # Ensure mask is also single-channel

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask > 0).float()  # Binarize the mask

        return image, mask


# Define image and mask transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
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


# UNet Model Definition
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoding path
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)

        # Decoding path with upsampling
        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)  # Match channels after concatenation
        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)  # Match channels after concatenation
        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)  # Match channels after concatenation
        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)  # Match channels after concatenation

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.functional.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(nn.functional.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(nn.functional.max_pool2d(enc3, kernel_size=2))
        bottleneck = self.bottleneck(nn.functional.max_pool2d(enc4, kernel_size=2))

        # Decoding path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)


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
model = UNet(in_channels=1, out_channels=2).to(device)

# Load previous weights if resuming training for fine-tuning
previous_weights_path = '../Model weights/best_segmentation_model_weights_unet.pth'
if os.path.exists(previous_weights_path):
    model.load_state_dict(torch.load(previous_weights_path))

# Criterion and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Ensure the save directory exists
save_dir = '../Model weights'
os.makedirs(save_dir, exist_ok=True)
best_weights_path = os.path.join(save_dir, 'best_segmentation_model_weights_unet.pth')

# Training and validation loop
num_epochs = 100

# Initialise variables
best_dice = -float('inf')
best_hausdorff = float('inf')

# Define a file path for saving metrics
csv_file_path = "../training_metrics_100epochs_unet.csv"
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

        # Forward pass and optimization
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

    # Compute average Dice and Hausdorff, with checks
    avg_val_loss = val_loss / len(val_loader)
    avg_dice = float(dice_total / len(val_loader.dataset))  # Average Dice across dataset
    avg_hausdorff = float(hausdorff_total / hausdorff_count) if hausdorff_count > 0 else float('inf')  # Only if non-infinite distances

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss}, Dice: {avg_dice}, Hausdorff: {avg_hausdorff}")

    # Log metrics for each epoch
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, avg_val_loss, avg_dice, avg_hausdorff])

    # Save weights if either Dice improves or Hausdorff decreases
    if avg_dice > best_dice or avg_hausdorff < best_hausdorff:
        best_dice = max(avg_dice, best_dice)
        best_hausdorff = min(avg_hausdorff, best_hausdorff)
        torch.save(model.state_dict(), best_weights_path)
        print(f"Best model saved with Dice: {avg_dice}, Hausdorff: {avg_hausdorff} at {best_weights_path}")

print(f"Metrics saved to {csv_file_path}")
print("--- Training complete: %s seconds ---" % (time.time() - start_time))