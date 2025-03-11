import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # For progress bar
import json


def compute_nonzero_stats_with_percentages(base_dir):
    results = {}

    # Iterate over pipe diameter folders
    for pipe_diameter_folder in os.listdir(base_dir):
        pipe_diameter_path = os.path.join(base_dir, pipe_diameter_folder)

        if not os.path.isdir(pipe_diameter_path):
            print(f"Skipping {pipe_diameter_folder}, not a directory.")
            continue

        masks_folder = os.path.join(pipe_diameter_path, "masks")
        if not os.path.exists(masks_folder):
            print(f"Skipping {masks_folder}, 'masks' folder not found.")
            continue

        mask_files = [f for f in os.listdir(masks_folder) if f.endswith(('.png', '.jpg', '.tif'))]
        nonzero_counts = []
        percentage_nonzero = []

        print(f"Processing masks in {pipe_diameter_folder}")
        for mask_file in tqdm(mask_files, desc=f"Processing masks in {pipe_diameter_folder}", leave=False):
            mask_path = os.path.join(masks_folder, mask_file)
            mask = np.array(Image.open(mask_path))

            # Count non-zero pixels
            nonzero_count = np.count_nonzero(mask)
            total_pixels = mask.size  # Total pixels in the mask
            nonzero_percentage = (nonzero_count / total_pixels) * 100

            nonzero_counts.append(nonzero_count)
            percentage_nonzero.append(nonzero_percentage)

        if nonzero_counts:
            avg_nonzero = np.mean(nonzero_counts)
            max_nonzero = np.max(nonzero_counts)
            min_nonzero = np.min(nonzero_counts)

            avg_percentage = np.mean(percentage_nonzero)
            max_percentage = np.max(percentage_nonzero)
            min_percentage = np.min(percentage_nonzero)

            # Store results
            results[pipe_diameter_folder] = {
                "Average Non-Zero Pixels": avg_nonzero,
                "Max Non-Zero Pixels": max_nonzero,
                "Min Non-Zero Pixels": min_nonzero,
                "Average Non-Zero Percentage": avg_percentage,
                "Max Non-Zero Percentage": max_percentage,
                "Min Non-Zero Percentage": min_percentage,
                "Total Masks Processed": len(nonzero_counts),
            }
        else:
            print(f"No valid masks found in {masks_folder}.")
            results[pipe_diameter_folder] = {
                "Average Non-Zero Pixels": None,
                "Max Non-Zero Pixels": None,
                "Min Non-Zero Pixels": None,
                "Average Non-Zero Percentage": None,
                "Max Non-Zero Percentage": None,
                "Min Non-Zero Percentage": None,
                "Total Masks Processed": 0,
            }

    return results


# Example usage
base_directory = "../Datasets/Lq e9 9L/Lq E9 9L - cropped"  # Adjust path as needed
pipe_stats = compute_nonzero_stats_with_percentages(base_directory)

# Print the results
for pipe, stats in pipe_stats.items():
    print(f"\nFolder: {pipe}")
    for stat, value in stats.items():
        print(f"  {stat}: {value}")

# Optionally save results to a JSON file
output_file = "nonzero_stats_with_percentages.json"
with open(output_file, "w") as f:
    json.dump(pipe_stats, f, indent=4)
print(f"\nResults saved to {output_file}")
