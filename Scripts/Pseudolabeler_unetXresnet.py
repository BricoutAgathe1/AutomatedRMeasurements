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
model.load_state_dict(torch.load('../Model weights/ChrisScanners_model_weights_unetXresnet18_split2.pth', map_location=torch.device('cpu')))
model.eval()


class UnlabeledUltrasoundDataset(Dataset):
    def __init__(self, image_dir, transform=None, exts=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        self.image_dir = image_dir
        all_entries = os.listdir(image_dir)
        self.image_filenames = sorted([
            f for f in all_entries
            if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(exts)
        ])
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
unlabeled_image_dir = '../Datasets/Chris_scanners/splits5/test/half_flipped'
pseudolabel_dir = '../Datasets/Chris_scanners/splits5/test/pseudolabels_MIApaper_split5'
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

import os
import csv
import cv2
import numpy as np
import math

# --- USER SETTINGS ---
output_csv_path = os.path.join(pseudolabel_dir, 'distance_measurements.csv')
conversion_csv_path = os.path.join(unlabeled_image_dir, 'conversion_factors.csv')

# --- Collect image files ---
image_files = sorted([
    f for f in os.listdir(unlabeled_image_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

# --- Load existing conversion factors if available ---
conversion_factors = {}
if os.path.exists(conversion_csv_path):
    print(f"Loading existing conversion factors from: {conversion_csv_path}")
    with open(conversion_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conversion_factors[row['Image Name']] = float(row['Conversion Factor (mm/pixel)'])
else:
    print("No existing conversion_factors.csv found — manual point selection required for all images.")

print("\nInstructions:")
print("- For each image WITHOUT a saved conversion factor:")
print("  • LEFT-CLICK twice to mark two points that are 10 mm apart.")
print("  • A green line will appear between the points.")
print("  • Press any key after selecting two points to continue.\n")

# --- Loop through original images ---
for idx, filename in enumerate(image_files, start=1):
    image_path = os.path.join(unlabeled_image_dir, filename)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))

    if img is None:
        print(f"Failed to load image {filename}")
        continue

    if filename not in conversion_factors:
        img_display = img.copy()
        points = []

        def select_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
                if len(points) == 2:
                    cv2.line(img_display, points[0], points[1], (0, 255, 0), 2)
                cv2.imshow("Select 10 mm distance", img_display)

        cv2.imshow("Select 10 mm distance", img_display)
        cv2.setMouseCallback("Select 10 mm distance", select_points)

        print(f"\n[{idx}] Select two points for {filename}...")
        while True:
            cv2.imshow("Select 10 mm distance", img_display)
            if len(points) == 2:
                break
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to skip
                print("Skipping image.")
                break

        cv2.destroyAllWindows()

        if len(points) < 2:
            continue

        (x1, y1), (x2, y2) = points
        pixel_distance = math.dist((x1, y1), (x1, y2))
        conversion_factor = 10.0 / pixel_distance  # mm/pixel
        conversion_factors[filename] = conversion_factor

        print(f"Saved {filename}: pixel_dist={pixel_distance:.2f}px, "
              f"conversion={conversion_factor:.9f} mm/px")
    else:
        print(f"[{idx}] Using saved conversion factor for {filename}")

# --- Save conversion_factors.csv ---
with open(conversion_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image Name', 'Conversion Factor (mm/pixel)'])
    for fname, cf in conversion_factors.items():
        writer.writerow([fname, cf])

print(f"\nAll conversion factors saved to: {conversion_csv_path}")

# --- Apply the saved conversion factors to pseudolabel masks ---
output_csv_path = os.path.join(pseudolabel_dir, 'distance_measurements.csv')
pseudolabel_files = sorted([f for f in os.listdir(pseudolabel_dir) if f.endswith('_pseudolabel.png')])

with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image Name', 'Conversion Factor (mm/pixel)', 'Top Row (px)', 'Bottom Row (px)', 'Top (mm)', 'Bottom (mm)'])

    for filename in pseudolabel_files:
        pseudolabel_path = os.path.join(pseudolabel_dir, filename)
        mask = cv2.imread(pseudolabel_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load pseudolabel for {filename}")
            continue

        # Match conversion factor by corresponding image name
        image_name = filename.replace('_pseudolabel.png', '.jpg').lower()
        conversion_factors_ci = {k.lower(): v for k, v in conversion_factors.items()}
        if image_name not in conversion_factors_ci:
            print(f"No conversion factor found for {image_name}, skipping.")
            continue

        conversion_factor = conversion_factors_ci[image_name]
        row_sums = np.sum(mask > 0, axis=1)
        non_zero_rows = np.where(row_sums > 0)[0]
        if len(non_zero_rows) == 0:
            print(f"{filename}: No non-zero pixels found.")
            continue

        top_row = int(non_zero_rows[0])
        bottom_row = int(non_zero_rows[-1])
        top_mm = top_row * conversion_factor
        bottom_mm = bottom_row * conversion_factor

        print(f"{filename}: conversion={conversion_factor:.9f} mm/px, "
              f"top={top_mm:.2f} mm, bottom={bottom_mm:.2f} mm")

        writer.writerow([filename, conversion_factor, top_row, bottom_row,
                         f"{top_mm:.2f}", f"{bottom_mm:.2f}"])

print(f"\nDistance measurements saved to: {output_csv_path}")
print("--- Distance measurements complete: %s seconds ---" % (time.time() - start_time))
cv2.destroyAllWindows()