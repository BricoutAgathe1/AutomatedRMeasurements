import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
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
            self.image_filenames.extend([
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if not f.lower().endswith('_mask.png')  # exclude masks
            ])

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        img_name = os.path.basename(img_path)

        # Safely create mask name regardless of image extension case
        basename, _ = os.path.splitext(img_name)
        mask_filename = basename + '_mask.png'

        mask_path = None
        for mask_dir in self.mask_dirs:
            mask_candidate = os.path.join(mask_dir, mask_filename)
            if os.path.exists(mask_candidate):
                mask_path = mask_candidate
                break

        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image: {img_path}")

        image = Image.open(img_path).convert("L")  # Convert image to grayscale
        mask = Image.open(mask_path).convert("L")  # Ensure mask is single-channel

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask > 0).float()

        return image, mask


# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=2).to(device)
model.load_state_dict(torch.load('../Model weights/ChrisScanners_model_weights_unetXresnet18_310725.pth', map_location=torch.device('cpu')))
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
unlabeled_image_dir = '../Datasets/Chris_scanners/Splits/test/half_flipped'
pseudolabel_dir = '../Datasets/Chris_scanners/Splits/test/pseudolabels_unetXresnet18'
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
            pseudolabel_name = os.path.splitext(image_name)[0] + '_pseudolabel.png'
            pseudolabel_np = (pseudolabel.cpu().numpy() * 255).astype(np.uint8)
            pseudolabel_path = os.path.join(pseudolabel_dir, pseudolabel_name)
            cv2.imwrite(pseudolabel_path, pseudolabel_np)


print("--- Pseudolabeling complete: %s seconds ---" % (time.time() - start_time))

import matplotlib.pyplot as plt
import csv

# === Distance Measurement ===
output_csv_path = os.path.join(pseudolabel_dir, 'distance_measurements.csv')
with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image Name', 'Conversion Factor (cm/pixel)', 'Top Row (px)', 'Bottom Row (px)', 'Top Row (cm)',
                     'Bottom Row (cm)'])

    print("\n--- Measuring distances for each image ---")
    for filename in sorted(os.listdir(pseudolabel_dir)):
        if not filename.endswith('_pseudolabel.png'):
            continue

        basename = filename.replace('_pseudolabel.png', '')
        possible_exts = ['.png', '.JPG']

        for ext in possible_exts:
            candidate_path = os.path.join(unlabeled_image_dir, basename + ext)
            if os.path.exists(candidate_path):
                original_img_path = candidate_path
                break

        if original_img_path is None:
            print(f"Original image not found for {filename}")
            continue

        # Step 1: Get conversion factor from original image
        original_img = cv2.imread(original_img_path)
        original_img = cv2.resize(original_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        print(f"\nImage: {filename}")
        plt.imshow(img_rgb)
        plt.title("Click two points to define a known real-world distance (in cm)")
        points = plt.ginput(2)
        plt.close()

        if len(points) != 2:
            print(f"Skipping {filename}: Did not get 2 points.")
            continue

        (y1, x1), (y2, x2) = points
        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        try:
            real_distance_mm = float(20)
        except ValueError:
            print(f"Invalid input. Skipping {filename}.")
            continue

        conversion_factor = real_distance_mm / pixel_distance

        # Step 2: Load pseudolabel and measure top/bottom extent
        pseudolabel_path = os.path.join(pseudolabel_dir, filename)
        mask = cv2.imread(pseudolabel_path, cv2.IMREAD_GRAYSCALE)
        row_sums = np.sum(mask > 0, axis=1)
        non_zero_rows = np.where(row_sums > 0)[0]

        if len(non_zero_rows) == 0:
            print(f"{filename}: No non-zero pixels found.")
            continue

        top_row = non_zero_rows[0]
        bottom_row = non_zero_rows[-1]

        top_mm = top_row * conversion_factor
        bottom_mm = bottom_row * conversion_factor

        print(f"{filename}:\n - Conversion: {conversion_factor:.4f} cm/pixel\n - Top: {top_mm:.2f} mm\n - Bottom: {bottom_mm:.2f} mm")

        writer.writerow([filename, conversion_factor, top_row, bottom_row, f"{top_mm:.2f}", f"{bottom_mm:.2f}"])