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
train_image_dir = 'data_cropped/train_cropped'
train_mask_dir = 'data_cropped/train_cropped'
val_image_dir = 'data_cropped/val_cropped'
val_mask_dir = 'data_cropped/val_cropped'
unlabeled_train_image_dir = 'data_cropped/train_unlabelled_cropped'

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

# unlabeled_image_dir = 'data/train_unlabeled'
unlabeled_image_dir = 'data_cropped/train_unlabelled_cropped'
unlabeled_dataset = UnlabeledUltrasoundDataset(unlabeled_image_dir, transform)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=False)

# pseudolabel_dir = 'data/train_pseudolabels'
pseudolabel_dir = 'data_cropped/train_pseudolabels_cropped'
os.makedirs(pseudolabel_dir, exist_ok=True)


def extract_top_bottom_positions(mask):
    # convert to grayscale if needed
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # find non-zero rows when mask is present
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
            # pseudolabel_img = pseudolabel_img.resize((1280, 720), Image.NEAREST) # uncomment if not using cropped data

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
                pseudolabel_path = os.path.join(pseudolabel_dir, image_name.replace('.jpg', '_pseudolabel.png'))
                pseudolabel_img.save(pseudolabel_path)
                distances.append({
                    'MaskFile': pseudolabel_path,
                    'DistanceFromTop': -1,  # Indicate no mask found
                    'DistanceFromBottom': -1  # Indicate no mask found
                })


print("--- Pseudolabeling complete: %s seconds ---" % (time.time() - start_time))


class DistanceInputApp:
    def __init__(self, root, image):
        self.root = root
        self.image = image
        self.canvas = Canvas(root, width=image.width, height=image.height)
        self.canvas.pack()
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.image_tk)
        self.canvas.bind("<Button-1>", self.on_click)
        self.points = []
        self.distance_entry = Entry(root)
        self.distance_entry.pack()
        self.label = Label(root, text="Enter the real-world distance in mm for the drawn line:")
        self.label.pack()
        self.button = Button(root, text="Submit", command=self.on_submit)
        self.button.pack()
        self.real_world_distance = None

    def on_click(self, event):
        if len(self.points) < 2:
            x, y = event.x, event.y
            self.points.append((x, y))
            if len(self.points) == 2:
                self.draw_line()

    def draw_line(self):
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        self.canvas.create_line(x1, y1, x2, y2, fill="red")

    def on_submit(self):
        self.real_world_distance = float(self.distance_entry.get())
        self.root.quit()

def get_distance_conversion_factor(image_path):
    image = Image.open(image_path).convert("RGB")
    root = Tk()
    app = DistanceInputApp(root, image)
    root.mainloop()

    if len(app.points) == 2 and app.real_world_distance is not None:
        (x1, y1), (x2, y2) = app.points
        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        conversion_factor = app.real_world_distance / pixel_distance
        return conversion_factor
    else:
        raise ValueError("Line drawing or distance input was not completed properly.")

# Define the conversion function
def convert_distances(distances, conversion_factor):
    for entry in distances:
        if entry['DistanceFromTop'] != -1:
            entry['DistanceFromTop'] *= conversion_factor
        if entry['DistanceFromBottom'] != -1:
            entry['DistanceFromBottom'] *= conversion_factor
    return distances


# Convert torch tensors to Python types
distances_serializable = []
for entry in distances:
    # Convert to Python int
    entry_serializable = {
        'MaskFile': entry['MaskFile'],
        'DistanceFromTop': int(entry['DistanceFromTop']),  # Convert to Python int
        'DistanceFromBottom': int(entry['DistanceFromBottom'])  # Convert to Python int
    }
    distances_serializable.append(entry_serializable)

with open('distances.json', 'w') as f:
    json.dump(distances_serializable, f)

print("--- Pseudolabeling complete: %s seconds ---" % (time.time() - start_time))

# Initialize variables to store max and min values
max_distance_from_top = float('-inf')
min_distance_from_top = float('inf')
max_distance_from_bottom = float('-inf')
min_distance_from_bottom = float('inf')

# Iterate through the distances list
for entry in distances:
    distance_from_top = entry['DistanceFromTop']
    distance_from_bottom = entry['DistanceFromBottom']

    # Update max and min values for DistanceFromTop if the distance is valid
    if distance_from_top != -1:
        if distance_from_top > max_distance_from_top:
            max_distance_from_top = distance_from_top
        if distance_from_top < min_distance_from_top:
            min_distance_from_top = distance_from_top

    # Update max and min values for DistanceFromBottom if the distance is valid
    if distance_from_bottom != -1:
        if distance_from_bottom > max_distance_from_bottom:
            max_distance_from_bottom = distance_from_bottom
        if distance_from_bottom < min_distance_from_bottom:
            min_distance_from_bottom = distance_from_bottom

# Check if no valid distances were found and set to None if so
if max_distance_from_top == float('-inf'):
    max_distance_from_top = None
if min_distance_from_top == float('inf'):
    min_distance_from_top = None
if max_distance_from_bottom == float('-inf'):
    max_distance_from_bottom = None
if min_distance_from_bottom == float('inf'):
    min_distance_from_bottom = None

print(f"Max Distance From Top: {max_distance_from_top} (pixels)")
print(f"Min Distance From Top: {min_distance_from_top} (pixels)")
print(f"Max Distance From Bottom: {max_distance_from_bottom} (pixels)")
print(f"Min Distance From Bottom: {min_distance_from_bottom} (pixels)")

# Get the first image path from the training dataset
first_image_path = os.path.join(train_image_dir, train_dataset.image_filenames[0])

# Get the pixel-to-mm conversion factor
conversion_factor = get_distance_conversion_factor(first_image_path)

# Load the distances data (assuming it's a list of dictionaries)
with open('distances.json', 'r') as f:
    distances = json.load(f)

# Convert distances from pixels to mm
distances_mm = convert_distances(distances, conversion_factor)

# Save the converted distances back to the JSON file
with open('distances_mm.json', 'w') as f:
    json.dump(distances_mm, f)

# Initialize variables to store max and min values
max_distance_from_top = float('-inf')
min_distance_from_top = float('inf')
max_distance_from_bottom = float('-inf')
min_distance_from_bottom = float('inf')

# Iterate through the distances list
for entry in distances_mm:
    distance_from_top = entry['DistanceFromTop']
    distance_from_bottom = entry['DistanceFromBottom']

    # Update max and min values for DistanceFromTop if the distance is valid
    if distance_from_top != -1:
        if distance_from_top > max_distance_from_top:
            max_distance_from_top = distance_from_top
        if distance_from_top < min_distance_from_top:
            min_distance_from_top = distance_from_top

    # Update max and min values for DistanceFromBottom if the distance is valid
    if distance_from_bottom != -1:
        if distance_from_bottom > max_distance_from_bottom:
            max_distance_from_bottom = distance_from_bottom
        if distance_from_bottom < min_distance_from_bottom:
            min_distance_from_bottom = distance_from_bottom

# Initialize variables to store total width and count of masks
total_width_mm = 0
num_masks = 0

# Angle of pipes to the vertical
angle_deg = 40
angle_rad = np.radians(angle_deg)

# Iterate through the distances list and calculate the width of each mask
for entry in distances:
    distance_from_top = entry['DistanceFromTop']
    distance_from_bottom = entry['DistanceFromBottom']

    # Update max and min values for DistanceFromTop if the distance is valid
    if distance_from_top != -1:
        if distance_from_top > max_distance_from_top:
            max_distance_from_top = distance_from_top
        if distance_from_top < min_distance_from_top:
            min_distance_from_top = distance_from_top

    # Update max and min values for DistanceFromBottom if the distance is valid
    if distance_from_bottom != -1:
        if distance_from_bottom > max_distance_from_bottom:
            max_distance_from_bottom = distance_from_bottom
        if distance_from_bottom < min_distance_from_bottom:
            min_distance_from_bottom = distance_from_bottom

    # Load the mask image
    mask_path = entry['mask_path']
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    # Calculate the width of the mask in pixels
    horizontal_sum = np.sum(mask_np, axis=0)
    width_pixels = np.sum(horizontal_sum > 0)

    # Adjust the width for the angle
    adjusted_width_pixels = width_pixels / np.cos(angle_rad)

    # Convert the width to mm
    width_mm = adjusted_width_pixels * conversion_factor

    # Update the total width and mask count
    total_width_mm += width_mm
    num_masks += 1

# Check if no valid distances were found and set to None if so
if max_distance_from_top == float('-inf'):
    max_distance_from_top = None
if min_distance_from_top == float('inf'):
    min_distance_from_top = None
if max_distance_from_bottom == float('-inf'):
    max_distance_from_bottom = None
if min_distance_from_bottom == float('inf'):
    min_distance_from_bottom = None

# Calculate the mean width
mean_width_mm = total_width_mm / num_masks if num_masks > 0 else None

print(f"Max Distance From Top: {max_distance_from_top} (mm)")
print(f"Min Distance From Top: {min_distance_from_top} (mm)")
print(f"Max Distance From Bottom: {max_distance_from_bottom} (mm)")
print(f"Min Distance From Bottom: {min_distance_from_bottom} (mm)")
print(f"Mean Width: {mean_width_mm} mm")