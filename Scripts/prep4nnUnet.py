import os
import numpy as np
from PIL import Image
import nibabel as nib
import json
from tqdm import tqdm

# === Set your paths ===
task_id = 999
task_name = f"Dataset{task_id}_TMPhantom"
base_output_dir = f"C:/Users/agath/nnUNet_data/nnUNet_raw/{task_name}"
input_image_dir = "C:/Users/agath/PycharmProjects/AutomatedRMeasurements/Datasets/nnU-Net data"
output_imagesTr = os.path.join(base_output_dir, "imagesTr")
output_labelsTr = os.path.join(base_output_dir, "labelsTr")

os.makedirs(output_imagesTr, exist_ok=True)
os.makedirs(output_labelsTr, exist_ok=True)

# === Resize resolution ===
target_size = (512, 512)

# === Helper function ===
def convert_to_nifti(input_path, output_path, is_mask=False):
    img = Image.open(input_path).convert("L")
    img = img.resize(target_size, Image.NEAREST if is_mask else Image.BILINEAR)

    arr = np.array(img, dtype=np.uint8)
    if is_mask:
        arr = (arr > 0).astype(np.uint8)  # Binarize the mask

        # Ensure the mask is formatted correctly (pipe is 1, background is 0)
        if np.max(arr) > 1:
            print(f"⚠️  Mask for {input_path} has values greater than 1, normalizing.")
            arr = arr // 255  # Normalize mask to 0 (background) and 1 (pipe)

        max_val = np.max(arr)

        # If the mask is empty, return None to skip this mask
        if max_val == 0:
            print(f"⚠️  Mask is empty for {input_path}, skipping.")
            return None  # Skip empty masks

    arr = arr[None, ...]  # Shape (1, H, W)

    nifti_img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(nifti_img, output_path)
    return output_path


# === Processing ===
image_files = sorted([f for f in os.listdir(input_image_dir) if f.endswith(".bmp")])
training_list = []

print("Converting and resizing images to 512x512...")
for idx, image_file in enumerate(tqdm(image_files)):
    base_id = f"TM_{idx:04d}"
    image_path = os.path.join(input_image_dir, image_file)
    mask_file = image_file.replace(".bmp", "_mask.png")
    mask_path = os.path.join(input_image_dir, mask_file)

    if not os.path.exists(mask_path):
        print(f"⚠️  Mask not found for {image_file}, skipping.")
        continue

    image_output_path = os.path.join(output_imagesTr, f"{base_id}_0000.nii.gz")
    mask_output_path = os.path.join(output_labelsTr, f"{base_id}.nii.gz")

    # Convert and save the image and mask
    image_path_out = convert_to_nifti(image_path, image_output_path, is_mask=False)
    mask_path_out = convert_to_nifti(mask_path, mask_output_path, is_mask=True)

    # Only add to the training list if both image and mask are valid
    if image_path_out and mask_path_out:
        training_list.append({
            "image": f"./imagesTr/{base_id}_0000.nii.gz",
            "label": f"./labelsTr/{base_id}.nii.gz"
        })

print(f"✅ Resized and converted {len(training_list)} image-mask pairs.")

# === Create dataset.json ===
json_dict = {
    "labels": {
        "0": "background",  # Background label should be 0
        "1": "pipe"
    },
    "channel_names": {
        "0": "US"
    },
    "file_ending": ".nii.gz",  # Ensures file ending for NIfTI files
    "numTraining": len(training_list),  # Number of training samples
    "numTest": 0,  # No test data in this case
    "num_classes": 2  # Number of classes (background + pipe)
}

# Save the dataset.json file
json_path = os.path.join(base_output_dir, "dataset.json")
with open(json_path, 'w') as f:
    json.dump(json_dict, f, indent=4)

print(f"✅ Created dataset.json at {json_path}")
print("🎉 All done! You can now preprocess with nnU-Net.")
