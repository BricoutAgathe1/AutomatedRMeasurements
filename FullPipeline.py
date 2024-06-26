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

start_time = time.time()


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
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
unlabeled_dataloader = DataLoader(unlabeled_train_dataset, batch_size=batch_size, shuffle=False)


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


# Example usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetSegmentation(backbone='resnet152', num_classes=2)  # Choose backbone as needed
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
num_epochs = 20
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device).long().squeeze(1)  # Ensure masks are of type long and remove extra channel dimension

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        outputs = nn.functional.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}")

    # Validation step
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
    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss}")

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_segmentation_model_weights.pth')
        print(f"Best model saved with val loss: {best_val_loss}")

print("--- Initial training complete:%s seconds ---" % (time.time() - start_time))

# Pseudolabel the unlabeled training data
model.load_state_dict(torch.load('best_segmentation_model_weights.pth'))
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

unlabeled_image_dir = 'data/train_unlabeled'
unlabeled_dataset = UnlabeledUltrasoundDataset(unlabeled_image_dir, transform)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=False)

pseudolabel_dir = 'data/train_pseudolabels'
os.makedirs(pseudolabel_dir, exist_ok=True)

def extract_top_bottom_positions(mask):
    # convert to grayscale if needed
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    #find non-zero rows when mask is present
    non_zero_rows = np.where(mask > 0)[0]

    if len(non_zero_rows) > 0:
        # first non-zero row
        top_position = non_zero_rows[0]
        # last non-zero row
        bottom_position = non_zero_rows[-1]
        return top_position, bottom_position
    else:
        return None, None # no mask found


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
            pseudolabel_img = pseudolabel_img.resize((1280, 720), Image.NEAREST)

            if np.any(pseudolabel_np):
                pseudolabel_path = os.path.join(pseudolabel_dir, image_name.replace('.jpg', '_pseudolabel.png'))
                pseudolabel_img.save(pseudolabel_path)
                print(f"Pseudolabel saved: {pseudolabel_path}")

                top_pos, bottom_pos = extract_top_bottom_positions(pseudolabel_np)
                if top_pos is not None and bottom_pos is not None:
                    image_height = pseudolabel_np.shape[0]
                    distance_from_top = top_pos
                    distance_from_bottom = image_height - bottom_pos - 1

                    distances.append({
                        'MaskFile': pseudolabel_path,
                        'DistanceFromTop': distance_from_top,
                        'DistanceFromBottom': distance_from_bottom
                    })
                else:
                    print(f"No mask found or mask is empty for {pseudolabel_path}")
            else:
                print(f"Warning: Empty pseudolabel for image {image_name}")

# Convert torch tensors to Python types
distances_serializable = []
for entry in distances:
    # Convert torch tensors to Python int or float
    entry_serializable = {
        'MaskFile': entry['MaskFile'],
        'DistanceFromTop': entry['DistanceFromTop'].item(),  # Convert torch int64 to Python int
        'DistanceFromBottom': entry['DistanceFromBottom'].item()  # Convert torch int64 to Python int
    }
    distances_serializable.append(entry_serializable)

with open('distances.json', 'w') as f:
    json.dump(distances_serializable, f)


print("--- Pseudolabeling complete: %s seconds ---" % (time.time() - start_time))


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

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0).float()  # Ensure binary values 0 and 1

        return image, mask


pseudolabeled_image_dir = 'data/train_unlabeled'
pseudolabeled_mask_dir = 'data/train_pseudolabels'
pseudolabeled_dataset = PseudolabeledUltrasoundDataset(pseudolabeled_image_dir, pseudolabeled_mask_dir,
                                                       image_transforms)

combined_dataset = CombinedUltrasoundDataset(train_dataset, pseudolabeled_dataset, image_transforms)
combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

# Continue training with combined dataset
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in combined_loader:
        images = images.to(device)
        masks = masks.to(device).long().squeeze(1)  # Ensure masks are of type long and remove extra channel dimension

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        outputs = nn.functional.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(combined_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}")

    # Validation step
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
    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss}")

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_combined_segmentation_model_weights.pth')
        print(f"Best model saved with val loss: {best_val_loss}")

print("--- Combined training complete:%s seconds ---" % (time.time() - start_time))
