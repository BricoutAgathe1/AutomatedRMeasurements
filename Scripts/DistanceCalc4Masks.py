import os
import csv
import cv2
import numpy as np
import math

# Pseudolabelling settings
unlabeled_image_dir = '../Datasets/Noisy pipes/Subtractive/MindrayA20_L33-8U/MindrayA20_L33-8U_Noise=0.7'
pseudolabel_dir = '../Datasets/Noisy pipes/Subtractive/MindrayA20_L33-8U/MindrayA20_L33-8U_Noise=0.7/Masks'
os.makedirs(pseudolabel_dir, exist_ok=True)

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
pseudolabel_files = sorted([f for f in os.listdir(pseudolabel_dir) if f.endswith('_mask.png')])

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
        image_name = filename.replace('_mask.png', '.png').lower()
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
cv2.destroyAllWindows()