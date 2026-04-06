import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image
from medpy.metric.binary import hd95 as hd, asd, precision, recall, ravd, jc
import csv
import scipy.stats as st
import time
from sklearn.model_selection import KFold
from torch.utils.data import Subset

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

        return image, mask, img_name


# Define image and mask transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


all_image_dirs = ['../Datasets/Chris_scanners/Merged_halfflipped']
all_mask_dirs = ['../Datasets/Chris_scanners/Merged_halfflipped']

full_dataset = UltrasoundDataset(
    all_image_dirs,
    all_mask_dirs,
    image_transform=transform,
    mask_transform=transform
)


def get_top_bottom(mask):
    mask_np = mask.cpu().numpy()

    coords = np.where(mask_np > 0)

    if len(coords[0]) == 0:
        return -1, -1  # no detection

    top = np.min(coords[0])     # smallest row index
    bottom = np.max(coords[0])  # largest row index

    return int(top), int(bottom)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
indices = np.arange(len(full_dataset))

# Define Dice coefficient function
def dice_score(pred, target):
    pred = pred.float()
    target = target.float()

    if pred.sum() == 0 and target.sum() == 0:
        return torch.tensor(1.0)

    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

# Define Hausdorff Distance function
def hausdorff_distance(pred, target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)

    # Check if both pred and target contain any binary object
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        # Return a large value or None to indicate an invalid distance
        return float('inf')  # Optionally, you could return `None` and handle it later

    return hd(pred_np, target_np)

def ASD(pred,target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        return 0.0
    return asd(pred_np, target_np)

def PRECISION(pred, target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        return 0.0
    return precision(pred_np, target_np)

def RECALL(pred, target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        return 0.0
    return recall(pred_np, target_np)

def RAVD(pred, target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        return 0.0
    return ravd(pred_np, target_np)

def JC(pred, target):
    pred_np = (pred.cpu().numpy() > 0).astype(int)
    target_np = (target.cpu().numpy() > 0).astype(int)
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        return 0.0
    return jc(pred_np, target_np)

# Training and validation loop
num_epochs = 50

fold_dice_scores = []
fold_hd_scores = []
fold_asd = []
fold_precision = []
fold_recall = []
fold_ravd = []
fold_jc = []

# Ensure the save directory exists
save_dir = '../Model weights'
os.makedirs(save_dir, exist_ok=True)
best_weights_path = os.path.join(save_dir, 'ChrisScanners_model_weights_unetXresnet18_Kfold.pth')

print(f"Dataset size: {len(full_dataset)}")

metrics_file = open('training_metrics.csv', mode='w', newline='')
metrics_writer = csv.writer(metrics_file)

metrics_writer.writerow([
    'fold', 'epoch', 'train_loss', 'val_loss',
    'dice', 'hd95', 'precision', 'recall',
    'asd', 'ravd', 'jc', 'is_best', 'epoch_time'
])

test_file = open('test_metrics.csv', mode='w', newline='')
test_writer = csv.writer(test_file)

test_writer.writerow([
    'fold', 'image_name', 'dice', 'hd95',
    'precision', 'recall', 'asd', 'ravd', 'jc',
    'top_pixel', 'bottom_pixel', 'inference_time'
])

summary_file = open('fold_summary.csv', 'w', newline='')
summary_writer = csv.writer(summary_file)

summary_writer.writerow([
    'fold', 'best_epoch', 'best_dice', 'best_hd95', 'training_time', 'avg_inference_time', 'fps'
])


for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
    print(f"\n===== FOLD {fold+1} =====")
    fold_start_time = time.time()

    # Split into train+val and test
    train_val_idx = train_idx
    test_idx = test_idx

    # Further split train into train + val
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.1875,  # ~15% overall
        random_state=42 + fold
    )

    # Create subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    best_dice = -float('inf')
    best_hausdorff = float('inf')
    best_epoch = -1
    best_model_path = f'../Model weights/fold_{fold}_best.pth'

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        # Training loop
        for images, masks, names in train_loader:
            images, masks = images.to(device), masks.to(device).long().squeeze(1)
            optimizer.zero_grad()

            # Forward pass and optimization
            outputs = model(images)
            outputs = nn.functional.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}")

        # Validation loop
        model.eval()
        val_loss, dice_total, hausdorff_total = 0.0, 0.0, 0.0
        hausdorff_count = 0
        precision_total, recall_total, asd_total, ravd_total, jc_total = 0, 0, 0, 0, 0
        count = 0
        with torch.no_grad():
            for images, masks, names in val_loader:
                images, masks = images.to(device), masks.to(device).long().squeeze(1)
                outputs = model(images)
                outputs = nn.functional.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
                val_loss += criterion(outputs, masks).item()

                preds = torch.argmax(outputs, dim=1)

                for pred, mask in zip(preds, masks):
                    dice_val = dice_score(pred, mask).item()
                    dice_total += dice_val
                    precision_val = PRECISION(pred, mask)
                    recall_val = RECALL(pred, mask)
                    asd_val = ASD(pred, mask)
                    ravd_val = RAVD(pred, mask)
                    jc_val = JC(pred, mask)
                    hausdorff_val = hausdorff_distance(pred, mask)

                    if hausdorff_val != float('inf'):
                        hausdorff_total += hausdorff_val
                        hausdorff_count += 1

                    precision_total += precision_val
                    recall_total += recall_val
                    asd_total += asd_val
                    ravd_total += ravd_val
                    jc_total += jc_val
                    count += 1

            # Compute average Dice and Hausdorff, with checks
            avg_val_loss = val_loss / len(val_loader)
            avg_dice = float(dice_total / len(val_loader.dataset))  # Average Dice across dataset
            avg_hausdorff = float(hausdorff_total / hausdorff_count) if hausdorff_count > 0 else float(
                'inf')  # Only if non-infinite distances
            avg_precision = precision_total / count
            avg_recall = recall_total / count
            avg_asd = asd_total / count
            avg_ravd = ravd_total / count
            avg_jc = jc_total / count

            scheduler.step(avg_dice)

            # Save weights if either Dice improves or Hausdorff decreases
            is_best = 0
            if avg_dice > best_dice:
                best_dice = avg_dice
                best_hausdorff = avg_hausdorff
                best_epoch = epoch
                is_best = 1
                torch.save(model.state_dict(), best_model_path)

            elif abs(avg_dice - best_dice) < 0.01 and avg_hausdorff < best_hausdorff:
                best_hausdorff = avg_hausdorff
                best_epoch = epoch
                is_best = 1
                torch.save(model.state_dict(), best_model_path)

            epoch_time = time.time() - epoch_start_time

            metrics_writer.writerow([
                fold,
                epoch,
                train_loss,
                avg_val_loss,
                avg_dice,
                avg_hausdorff,
                avg_precision,
                avg_recall,
                avg_asd,
                avg_ravd,
                avg_jc,
                is_best,
                epoch_time
            ])


    fold_training_time = time.time() - fold_start_time
    print(f"Fold {fold + 1} training time: {fold_training_time:.2f} seconds")

    # Testing loop
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    dice_total, hausdorff_total, precision_total, recall_total, asd_total, ravd_total, jc_total = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dice_count, hd_count, precision_count, recall_count, asd_count, ravd_count, jc_count = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_inference_time = 0

    with torch.no_grad():
        for images, masks, names in test_loader:
            images, masks = images.to(device), masks.to(device).long().squeeze(1)
            for i in range(images.shape[0]):
                img = images[i].unsqueeze(0)
                mask = masks[i]
                name = names[i]

                start_inf = time.time()
                output = model(img)
                output = nn.functional.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
                pred = torch.argmax(output, dim=1).squeeze(0)

                inference_time = time.time() - start_inf
                total_inference_time += inference_time

                dice_val = dice_score(pred, mask).item()
                hd_val = hausdorff_distance(pred, mask)
                precision_val = PRECISION(pred, mask)
                recall_val = RECALL(pred, mask)
                asd_val = ASD(pred, mask)
                ravd_val = RAVD(pred, mask)
                jc_val = JC(pred, mask)

                top, bottom = get_top_bottom(pred)

                dice_total += dice_val
                precision_total += precision_val
                recall_total += recall_val
                asd_total += asd_val
                ravd_total += ravd_val
                jc_total += jc_val
                dice_count += 1
                precision_count += 1
                recall_count += 1
                asd_count += 1
                ravd_count += 1
                jc_count += 1
                if hd_val != float('inf'):
                    hausdorff_total += hd_val
                    hd_count += 1

                test_writer.writerow([
                    fold,
                    name,
                    dice_val,
                    hd_val,
                    precision_val,
                    recall_val,
                    asd_val,
                    ravd_val,
                    jc_val,
                    top,
                    bottom,
                    inference_time
                ])

        avg_inference_time = total_inference_time / dice_count
        fps = 1/avg_inference_time
        avg_dice = dice_total / dice_count if dice_count > 0 else 0.0
        avg_hd = hausdorff_total / hd_count if hd_count > 0 else float('inf')
        avg_precision = precision_total / precision_count if precision_count > 0 else 0.0
        avg_recall = recall_total / recall_count if recall_count > 0 else 0.0
        avg_asd = asd_total / asd_count if asd_count > 0 else 0.0
        avg_ravd = ravd_total / ravd_count if ravd_count > 0 else 0.0
        avg_jc = jc_total / jc_count if jc_count > 0 else 0.0
        hd_failure_rate = 1 - (hd_count / dice_count)
        print(f"HD95 failure rate: {hd_failure_rate:.2%}")

    fold_dice_scores.append(avg_dice)
    fold_hd_scores.append(avg_hd)
    fold_precision.append(avg_precision)
    fold_recall.append(avg_recall)
    fold_asd.append(avg_asd)
    fold_ravd.append(avg_ravd)
    fold_jc.append(avg_jc)
    print(f"Fold {fold + 1} → Dice: {avg_dice:.4f}, HD95: {avg_hd:.2f}, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, ASD: {avg_asd:.2f}, RAVD: {avg_ravd:.2f}, JC: {avg_jc:.2f}")

    summary_writer.writerow([
        fold,
        best_epoch,
        best_dice,
        best_hausdorff,
        fold_training_time,
        avg_inference_time,
        fps
    ])

metrics_file.close()
test_file.close()
summary_file.close()
print("--- Training & testing complete: %s seconds ---" % (time.time() - start_time))
print("\n===== FINAL K-FOLD RESULTS =====")

n_dice = len(fold_dice_scores)
n_hd = len(fold_hd_scores)
n_precision = len(fold_precision)
n_recall = len(fold_recall)
n_asd = len(fold_asd)
n_ravd = len(fold_ravd)
n_jc = len(fold_jc)
t_dice = st.t.ppf(0.975, df=n_dice-1)
t_hd = st.t.ppf(0.975, df=n_hd-1)
t_precision = st.t.ppf(0.975, df=n_precision-1)
t_recall = st.t.ppf(0.975, df=n_recall-1)
t_asd = st.t.ppf(0.975, df=n_asd-1)
t_ravd = st.t.ppf(0.975, df=n_ravd-1)
t_jc = st.t.ppf(0.975, df=n_jc-1)
dice_ci = t_dice * np.std(fold_dice_scores) / np.sqrt(n_dice)
hd_ci = t_hd * np.std(fold_hd_scores) / np.sqrt(n_hd)
precision_ci = t_precision * np.std(fold_precision) / np.sqrt(n_precision)
recall_ci = t_recall * np.std(fold_recall) / np.sqrt(n_recall)
asd_ci = t_asd * np.std(fold_asd) / np.sqrt(n_asd)
ravd_ci = t_ravd * np.std(fold_ravd) / np.sqrt(n_ravd)
jc_ci = t_jc * np.std(fold_jc) / np.sqrt(n_jc)

print(f"Dice: {np.mean(fold_dice_scores):.4f} ± {dice_ci:.4f}")
print(f"HD95: {np.mean(fold_hd_scores):.2f} ± {hd_ci:.2f}")
print(f"Precision: {np.mean(fold_precision):.4f} ± {precision_ci:.4f}")
print(f"Recall: {np.mean(fold_recall):.4f} ± {recall_ci:.4f}")
print(f"ASD: {np.mean(fold_asd):.2f} ± {asd_ci:.2f}")
print(f"RAVD: {np.mean(fold_ravd):.2f} ± {ravd_ci:.2f}")
print(f"JC: {np.mean(fold_jc):.4f} ± {jc_ci:.4f}")