import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time

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
        mask_name = img_name.replace('.jpg', '_mask.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0).float()  # Ensure binary values 0 and 1

        return image, mask


# Define image and mask transformations
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Paths to datasets
train_image_dir = '../Datasets/Testing/Global_Dataset/train/cropped_images'
train_mask_dir = '../Datasets/Testing/Global_Dataset/train/cropped_images'
val_image_dir = '../Datasets/Testing/Global_Dataset/val/cropped_images'
val_mask_dir = '../Datasets/Testing/Global_Dataset/val/cropped_images'
unlabeled_train_image_dir = '../Datasets/Testing/Global_Dataset/Pseudolabeled'

# Initialize datasets
train_dataset = UltrasoundDataset(train_image_dir, train_mask_dir, image_transforms)
val_dataset = UltrasoundDataset(val_image_dir, val_mask_dir, image_transforms)
unlabeled_train_dataset = UltrasoundDataset(unlabeled_train_image_dir, train_mask_dir, image_transforms)

# Create DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
unlabeled_dataloader = DataLoader(unlabeled_train_dataset, batch_size=batch_size, shuffle=False)


# Define the UNet segmentation model
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


# Combine manually labeled and pseudolabeled data
class CombinedUltrasoundDataset(Dataset):
    def __init__(self, labeled_dataset, pseudolabeled_dataset, transform=None):
        self.labeled_dataset = labeled_dataset
        self.pseudolabeled_dataset = pseudolabeled_dataset
        self.transform = transform

    def __len__(self):
        return len(self.labeled_dataset) + len(self.pseudolabeled_dataset)

    def __getitem__(self, idx):
        if idx < len(self.labeled_dataset):
            return self.labeled_dataset[idx]
        else:
            return self.pseudolabeled_dataset[idx - len(self.labeled_dataset)]


# Load pseudolabeled dataset
class PseudolabeledUltrasoundDataset(Dataset):
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
        mask_name = img_name.replace('.jpg', '_pseudolabel.png')  # Ensure correct mask path
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0).float()  # Ensure binary values 0 and 1

        return image, mask


pseudolabeled_image_dir = '../Datasets/Testing/Global_Dataset/ToPseudolabel'
pseudolabeled_mask_dir = '../Datasets/Testing/Global_Dataset/Pseudolabeled'
pseudolabeled_dataset = PseudolabeledUltrasoundDataset(pseudolabeled_image_dir, pseudolabeled_mask_dir, image_transforms)

combined_dataset = CombinedUltrasoundDataset(train_dataset, pseudolabeled_dataset, image_transforms)
combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=2).to(device)

# Load previous weights if resuming training
previous_weights_path = '../Model weights/best_segmentation_model_weights_unet.pth'
if os.path.exists(previous_weights_path):
    model.load_state_dict(torch.load(previous_weights_path))

# Criterion and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Define Dice coefficient function
def dice_score(pred, target):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

# Ensure the save directory exists
save_dir = '../Model weights'
os.makedirs(save_dir, exist_ok=True)
best_weights_path = os.path.join(save_dir, 'best_combined_segmentation_model_weights_unet.pth')

# Training loop as before, using the combined dataset loader

# Training and validation loop
num_epochs = 20

best_dice = -float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Training loop
    for images, masks in combined_loader:
        images, masks = images.to(device), masks.to(device).long().squeeze(1)
        optimizer.zero_grad()

        # Forward pass and optimization
        outputs = model(images)
        outputs = nn.functional.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}")

    # Validation loop
    model.eval()
    val_loss, dice_total = 0.0, 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device).long().squeeze(1)
            outputs = model(images)
            outputs = nn.functional.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
            val_loss += criterion(outputs, masks).item()

            preds = torch.argmax(outputs, dim=1)

            for pred, mask in zip(preds, masks):
                dice_total += dice_score(pred, mask)

    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)
    avg_dice = dice_total / len(val_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss}, Dice: {avg_dice}")

    # Update score and save model if improved
    if avg_dice > best_dice:
        best_dice = avg_dice
        torch.save(model.state_dict(), best_weights_path)
        print(f"Best model saved with Dice score: {best_dice} at {best_weights_path}")

print("--- Combined training complete: %s seconds ---" % (time.time() - start_time))
