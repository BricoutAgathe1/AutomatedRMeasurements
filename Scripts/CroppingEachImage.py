import cv2
import os
import numpy as np

# Global state
ref_point = []
cropping = False
image = None
clone = None

# Constants
RESIZE_DIM = (256, 256)

def safe_crop_and_resize(img, ref_point):
    x1, y1 = ref_point[0]
    x2, y2 = ref_point[1]
    side_length = x2 - x1

    h, w = img.shape[:2]
    x_end = x1 + side_length
    y_end = y1 + side_length

    # Calculate out-of-bound paddings
    left_pad = max(0, -x1)
    top_pad = max(0, -y1)
    right_pad = max(0, x_end - w)
    bottom_pad = max(0, y_end - h)

    # Clamp crop coords
    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(w, x_end)
    y2_clamped = min(h, y_end)

    # Crop
    cropped = img[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

    # Pad
    if any(p > 0 for p in [left_pad, top_pad, right_pad, bottom_pad]):
        cropped = cv2.copyMakeBorder(
            cropped, top_pad, bottom_pad, left_pad, right_pad,
            borderType=cv2.BORDER_CONSTANT, value=0
        )

    return cv2.resize(cropped, RESIZE_DIM, interpolation=cv2.INTER_AREA)

def draw_square(event, x, y, flags, param):
    global ref_point, cropping, image, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point.clear()
        ref_point.append((x, y))
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        temp = clone.copy()
        side_length = max(abs(x - ref_point[0][0]), abs(y - ref_point[0][1]))
        end_point = (ref_point[0][0] + side_length, ref_point[0][1] + side_length)
        cv2.rectangle(temp, ref_point[0], end_point, (0, 255, 0), 2)
        cv2.imshow("image", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        side_length = max(abs(x - ref_point[0][0]), abs(y - ref_point[0][1]))
        end_point = (ref_point[0][0] + side_length, ref_point[0][1] + side_length)
        ref_point.append(end_point)
        cropping = False
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

# Directories
input_dir = "../Datasets/Chris_scanners/Mindray A20/SLM10-3U"
output_dir_img = os.path.join(input_dir, "cropped")
output_dir_mask = os.path.join(input_dir, "cropped_masks")
os.makedirs(output_dir_img, exist_ok=True)
os.makedirs(output_dir_mask, exist_ok=True)

# Files to process
image_files = [f for f in os.listdir(input_dir) if f.endswith((".png", ".JPG", ".bmp", ".jpg")) and not f.endswith("_mask.png")]

# Loop over images
for filename in image_files:
    redo = True
    while redo:
        ref_point = []
        img_path = os.path.join(input_dir, filename)
        mask_path = os.path.join(input_dir, filename.rsplit('.', 1)[0] + "_mask.png")

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if image is None or mask is None:
            print(f"Skipping {filename} — image or mask not found or unreadable.")
            break

        clone = image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_square)

        print(f"\nCropping: {filename}")
        print("→ Draw a square with your mouse.")
        print("→ Press 'c' to confirm and crop.")
        print("→ Press 'r' to redo the selection.")
        print("→ Press 'q' to quit.")

        while True:
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c") and len(ref_point) == 2:
                cropped_img = safe_crop_and_resize(clone, ref_point)
                cropped_mask = safe_crop_and_resize(mask, ref_point)

                cv2.imwrite(os.path.join(output_dir_img, filename), cropped_img)
                cv2.imwrite(os.path.join(output_dir_mask, os.path.basename(mask_path)), cropped_mask)

                print(f"✓ Saved: {filename} and its mask")
                redo = False
                break

            elif key == ord("r"):
                print("↻ Redoing crop...")
                image = clone.copy()
                break

            elif key == ord("q"):
                print("🛑 Exiting early.")
                cv2.destroyAllWindows()
                exit()

        cv2.destroyAllWindows()

print("✅ All images processed.")
