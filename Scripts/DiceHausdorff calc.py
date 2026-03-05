import os
import csv
import torch
import numpy as np
import cv2
from medpy.metric.binary import hd95, hd, jc, precision, assd

# Paths
gt_folder = '../Datasets/Chris_scanners/Splits5/test/masks/half_flipped'
pseudo_folder = '../Datasets/Chris_scanners/Splits5/test/pseudolabels_MIApaper_split5'
output_csv = '../Datasets/Chris_scanners/Splits5/test/pseudolabels_MIApaper_split5/Full_metrics.csv'

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

def hausdorff_distance95(pred, target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        return float('inf')
    return hd95(pred_np, target_np)

def jaccard(pred, target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        return 1.0
    return jc(pred_np, target_np)

def precision_score(pred, target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)
    if np.sum(pred_np) == 0:
        return 1.0
    return precision(pred_np, target_np)

def assd_score(pred, target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        return float('inf')
    return assd(pred_np, target_np)

# Evaluation loop
dice_scores = []
hausdorff_scores = []
hausdorff95_scores = []
jaccard_coeffs = []
precision_scores = []
assd_scores = []

with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'Dice Score', 'Hausdorff Distance', 'Hausdorff 95 Distance', 'Jaccard Index', 'Precision', 'ASSD'])

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
            hausdorff95 = hausdorff_distance95(pseudo_tensor, gt_tensor)
            jaccard_idx = jaccard(pseudo_tensor, gt_tensor)
            precision_metric = precision_score(pseudo_tensor, gt_tensor)
            assd_metric = assd_score(pseudo_tensor, gt_tensor)

            dice_scores.append(dice)
            jaccard_coeffs.append(jaccard_idx)
            precision_scores.append(precision_metric)
            assd_scores.append(assd_metric)

            if hausdorff != float('inf'):
                hausdorff_scores.append(hausdorff)
            if hausdorff95 != float('inf'):
                hausdorff95_scores.append(hausdorff95)

            writer.writerow([fname, dice, hausdorff, hausdorff95, jaccard_idx, precision_metric, assd_metric])
            print(f"{fname}: Dice = {dice:.4f}, Hausdorff = {hausdorff:.2f}, Hausdorff95 = {hausdorff95:.2f}, Jaccard = {jaccard_idx:.4f}, Precision = {precision_metric:.4f}, ASSD = {assd_metric:.2f}")

# Summary
mean_dice = np.mean(dice_scores)
mean_hd = np.mean(hausdorff_scores) if hausdorff_scores else float('inf')
mean_hd95 = np.mean(hausdorff95_scores) if hausdorff95_scores else float('inf')
mean_jac = np.mean(jaccard_coeffs) if jaccard_coeffs else float('inf')
mean_precision = np.mean(precision_scores) if precision_scores else float('inf')
mean_assd = np.mean(assd_scores) if assd_scores else float('inf')

print(f"\n✅ Average Dice Score: {mean_dice:.4f}")
print(f"✅ Average Hausdorff Distance: {mean_hd:.2f}")
print(f"✅ Average Hausdorff 95 Distance: {mean_hd95:.2f}")
print(f"✅ Average Jaccard Index: {mean_jac:.4f}")
print(f"✅ Average Precision: {mean_precision:.4f}")
print(f"✅ Average ASSD: {mean_assd:.2f}")
print(f"📄 Metrics saved to: {output_csv}")