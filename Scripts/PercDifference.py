import json
import os
import numpy as np
import cv2  # Assuming OpenCV is used for image processing


def extract_top_bottom_positions(mask):
    """Extracts the topmost and bottommost pixel positions in a binary mask."""
    mask_indices = np.where(mask > 0)  # Get nonzero pixel indices
    if mask_indices[0].size > 0:
        top_pos = np.min(mask_indices[0])  # Topmost row with segmentation
        bottom_pos = np.max(mask_indices[0])  # Bottommost row with segmentation
        return top_pos, bottom_pos
    return None, None

# Paths to manual and ML masks
manual_mask_dir = "../Datasets/TM_Split/test/cropped_masks_all"
ml_mask_dir = "../Datasets/TM_Split/test/UnetResnet18_reproducibility/run2"

# Get list of manual mask files and infer corresponding pseudolabel filenames
manual_mask_files = [f for f in os.listdir(manual_mask_dir) if f.endswith("_mask.png")]

distances = []
all_percentage_diffs = []  # Store all top & bottom diffs together

for manual_file in manual_mask_files:
    # Derive the corresponding pseudolabel filename
    base_name = manual_file.replace("_mask.png", "")
    ml_file = f"{base_name}_pseudolabel.png"

    manual_path = os.path.join(manual_mask_dir, manual_file)
    ml_path = os.path.join(ml_mask_dir, ml_file)

    if os.path.exists(ml_path):
        # Read masks as grayscale
        manual_mask = cv2.imread(manual_path, cv2.IMREAD_GRAYSCALE)
        manual_mask = cv2.resize(manual_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        ml_mask = cv2.imread(ml_path, cv2.IMREAD_GRAYSCALE)

        # Extract top and bottom positions
        top_manual, bottom_manual = extract_top_bottom_positions(manual_mask)
        top_ml, bottom_ml = extract_top_bottom_positions(ml_mask)

        # Compute percentage differences for top and bottom distances
        top_diff, bottom_diff = None, None

        if top_manual is not None and top_ml is not None and top_manual > 0:
            top_diff = abs(top_manual - top_ml) / top_manual * 100
            all_percentage_diffs.append(top_diff)  # Store in overall list

        if bottom_manual is not None and bottom_ml is not None and bottom_manual > 0:
            bottom_diff = abs(bottom_manual - bottom_ml) / bottom_manual * 100
            all_percentage_diffs.append(bottom_diff)  # Store in overall list

        distances.append({
            "MaskFile": base_name,
            "ManualTop": int(top_manual) if top_manual is not None else -1,
            "ManualBottom": int(bottom_manual) if bottom_manual is not None else -1,
            "MLTop": int(top_ml) if top_ml is not None else -1,
            "MLBottom": int(bottom_ml) if bottom_ml is not None else -1,
            "TopPercentageDifference": top_diff if top_diff is not None else None,
            "BottomPercentageDifference": bottom_diff if bottom_diff is not None else None
        })

# Save distances and percentage differences to a JSON file
with open('../Distances/computed_distances.json', 'w') as f:
    json.dump(distances, f, indent=4)

# Compute **one overall** percentage difference and standard deviation
overall_percentage_diff = np.mean(all_percentage_diffs) if all_percentage_diffs else None
std_percentage_diff = np.std(all_percentage_diffs) if all_percentage_diffs else None

print(f"Overall Average Percentage Difference: {overall_percentage_diff:.2f}%")
print(f"Standard Deviation of Percentage Differences: {std_percentage_diff:.2f}%")