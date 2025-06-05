import cv2
import numpy as np
import os
import glob

# Directory containing images and masks
image_dir = ("../Datasets/Elegra_JPG/cropped")

# Get all .bmp images
image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

for image_path in image_files:
    # Construct mask filename
    mask_path = image_path.replace(".jpg", "_pseudolabel.png")

    # Check if corresponding mask exists
    if not os.path.exists(mask_path):
        print(f"Skipping {image_path} (no corresponding mask found)")
        continue

    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert grayscale mask to a 3-channel red overlay
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 2] = mask  # Red channel

    # Overlay the mask on the original image
    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

    # Show the overlay
    cv2.imshow("Overlay", overlay)

    # Wait for key press before showing next image
    key = cv2.waitKey(0)
    if key == 27:  # Press 'Esc' to exit early
        break

cv2.destroyAllWindows()
