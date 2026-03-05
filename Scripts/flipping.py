import os
import cv2
import random
from pathlib import Path

# Paths for the image and mask directories
train_images_folder = '../Datasets/Chris_scanners/Splits3/test/images'
train_masks_folder = '../Datasets/Chris_scanners/Splits3/test/masks'
augmented_images_folder = '../Datasets/Chris_scanners/Splits3/test/half_flipped'
augmented_masks_folder = '../Datasets/Chris_scanners/Splits3/test/half_flipped'

# Create augmented folders if they don’t exist
os.makedirs(augmented_images_folder, exist_ok=True)
os.makedirs(augmented_masks_folder, exist_ok=True)

# Get a list of images and their corresponding masks
images = sorted([f for f in os.listdir(train_images_folder) if f.endswith(('.jpg', '.JPG'))])
masks = sorted([f for f in os.listdir(train_masks_folder) if f.endswith('_mask.png')])

# Randomly shuffle to ensure random selection for flipping
random.shuffle(images)

# Split the list in half for flipping
flip_count = len(images) // 2
images_to_flip = images[:flip_count]
images_to_keep = images[flip_count:]

# Process each image and mask
for image_name in images:
    # Corresponding mask name (extension-safe)
    mask_name = os.path.splitext(image_name)[0] + '_mask.png'

    # Full paths
    image_path = os.path.join(train_images_folder, image_name)
    mask_path = os.path.join(train_masks_folder, mask_name)

    # Read the image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    if image is None or mask is None:
        print(f"Skipping {image_name} (missing image or mask)")
        continue

    # Flip or keep as is
    if image_name in images_to_flip:
        flipped_image = cv2.flip(image, 1)
        flipped_mask = cv2.flip(mask, 1)
        cv2.imwrite(os.path.join(augmented_images_folder, image_name), flipped_image)
        cv2.imwrite(os.path.join(augmented_masks_folder, mask_name), flipped_mask)
    else:
        cv2.imwrite(os.path.join(augmented_images_folder, image_name), image)
        cv2.imwrite(os.path.join(augmented_masks_folder, mask_name), mask)

print("Image and mask augmentation completed.")
