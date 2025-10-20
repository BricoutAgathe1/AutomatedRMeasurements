import os
import random
import shutil

# Define paths
parent_directory = "../Datasets/Chris_scanners/Cropped"  # Directory containing all folders
merged_folder = "../Datasets/Chris_scanners/Merged"  # Destination folder for all images and masks
output_dir = "../Datasets/Chris_scanners/Splits"  # Output folder for dataset splits

# Create merged folder if it doesn't exist
os.makedirs(merged_folder, exist_ok=True)

# Step 1: Copy only images that have a corresponding mask
for folder in os.listdir(parent_directory):
    folder_path = os.path.join(parent_directory, folder)

    # Ensure it's a directory
    if os.path.isdir(folder_path):
        images_in_folder = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".JPG"))]

        for img_file in images_in_folder:
            mask_file = os.path.splitext(img_file)[0] + "_mask.png"
            mask_path = os.path.join(folder_path, mask_file)

            # Check if corresponding mask exists
            if os.path.exists(mask_path):
                img_source = os.path.join(folder_path, img_file)
                mask_source = mask_path

                # Rename files using the folder name as a prefix
                new_img_name = f"{folder}_{img_file}"
                new_mask_name = f"{folder}_{mask_file}"

                img_dest = os.path.join(merged_folder, new_img_name)
                mask_dest = os.path.join(merged_folder, new_mask_name)

                shutil.copy(img_source, img_dest)  # Copy image
                shutil.copy(mask_source, mask_dest)  # Copy mask
            else:
                print(f"Skipping {img_file} (No matching mask found)")

print("All valid image-mask pairs have been copied to the merged directory.")

# Step 2: Split dataset into train, val, and test
# Create output directories
split_names = ["train", "val", "test"]
for split in split_names:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

# Collect image-mask pairs
images = [f for f in os.listdir(merged_folder) if f.endswith((".jpg", ".JPG"))]
random.shuffle(images)  # Shuffle randomly

# Define split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Compute split sizes
total_images = len(images)
train_size = int(total_images * train_ratio)
val_size = int(total_images * val_ratio)

# Assign images to splits
train_images = images[:train_size]
val_images = images[train_size:train_size + val_size]
test_images = images[train_size + val_size:]


# Function to copy images and masks to respective datasets
def copy_files(image_list, split_name):
    for img_file in image_list:
        mask_file = os.path.splitext(img_file)[0] + "_mask.png"
        img_src = os.path.join(merged_folder, img_file)
        mask_src = os.path.join(merged_folder, mask_file)

        img_dest = os.path.join(output_dir, split_name, "images", img_file)
        mask_dest = os.path.join(output_dir, split_name, "masks", mask_file)

        shutil.copy(img_src, img_dest)
        shutil.copy(mask_src, mask_dest)


# Copy files into dataset splits
copy_files(train_images, "train")
copy_files(val_images, "val")
copy_files(test_images, "test")

print("Dataset successfully copied into training, validation, and test sets!")
