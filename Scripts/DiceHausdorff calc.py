import os
import csv
import torch
import numpy as np
import cv2
from medpy.metric.binary import hd

# Paths
gt_folder = '../Datasets/Chris_scanners/Splits/test/masks'
pseudo_folder = '../Datasets/Chris_scanners/Splits/test/pseudolabels_unetXresnet18'
output_csv = '../Datasets/Chris_scanners/Splits/test/pseudolabels_unetXresnet18/Hausdorff_Dice_metrics.csv'

# Dice and Hausdorff functions
def dice_score(pred, target):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    if pred.sum() + target.sum() == 0:
        return 1.0  # Both empty, perfect match
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

def hausdorff_distance(pred, target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        return float('inf')
    return hd(pred_np, target_np)

# Evaluation loop
dice_scores = []
hausdorff_scores = []

with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'Dice Score', 'Hausdorff Distance'])

    for fname in os.listdir(gt_folder):
        if fname.endswith('_mask.png'):
            gt_path = os.path.join(gt_folder, fname)
            pseudo_name = fname.replace('_mask.png', '_pseudolabel.png')
            pseudo_path = os.path.join(pseudo_folder, pseudo_name)

            if not os.path.exists(pseudo_path):
                print(f"Skipping {fname}: pseudolabel not found.")
                continue

            # Read and preprocess
            gt_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pseudo_np = cv2.imread(pseudo_path, cv2.IMREAD_GRAYSCALE)

            if gt_np.shape != pseudo_np.shape:
                pseudo_np = cv2.resize(pseudo_np, gt_np.shape[::-1], interpolation=cv2.INTER_NEAREST)

            gt_tensor = torch.from_numpy((gt_np > 0).astype(np.uint8))
            pseudo_tensor = torch.from_numpy((pseudo_np > 0).astype(np.uint8))

            # Compute metrics
            dice = dice_score(pseudo_tensor, gt_tensor).item()
            hausdorff = hausdorff_distance(pseudo_tensor, gt_tensor)

            dice_scores.append(dice)
            if hausdorff != float('inf'):
                hausdorff_scores.append(hausdorff)

            writer.writerow([fname, dice, hausdorff])
            print(f"{fname}: Dice = {dice:.4f}, Hausdorff = {hausdorff:.2f}")

# Summary
mean_dice = np.mean(dice_scores)
mean_hd = np.mean(hausdorff_scores) if hausdorff_scores else float('inf')

print(f"\n✅ Average Dice Score: {mean_dice:.4f}")
print(f"✅ Average Hausdorff Distance: {mean_hd:.2f}")
print(f"📄 Metrics saved to: {output_csv}")