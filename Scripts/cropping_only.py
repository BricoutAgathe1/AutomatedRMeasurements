import cv2
import os

# Initialize global variables
ref_point = []
cropping = False
image = None

# Function to crop images
def crop_image(img, ref_point):
    x_start = ref_point[0][0]
    y_start = ref_point[0][1]
    side_length = ref_point[1][0] - ref_point[0][0]
    cropped_img = img[y_start:y_start + side_length, x_start:x_start + side_length]
    return cropped_img

# Function to process multiple images based on selected crop area
def crop_images(input_dir, output_dir, ref_point):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp") or filename.endswith(".JPG"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip if the image is not loaded properly
            cropped_img = crop_image(img, ref_point)
            cropped_img = cv2.resize(cropped_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, cropped_img)
    print(f"Cropped images saved in: {output_dir}")

# Function to handle mouse events
def draw_square(event, x, y, flags, param):
    global ref_point, cropping, image

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            image_copy = image.copy()
            side_length = max(abs(x - ref_point[0][0]), abs(y - ref_point[0][1]))
            end_point = (ref_point[0][0] + side_length, ref_point[0][1] + side_length)
            cv2.rectangle(image_copy, ref_point[0], end_point, (0, 255, 0), 2)
            cv2.imshow("image", image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        side_length = max(abs(x - ref_point[0][0]), abs(y - ref_point[0][1]))
        end_point = (ref_point[0][0] + side_length, ref_point[0][1] + side_length)
        ref_point[1] = end_point
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

# Load an example image to select the crop area
image_path = "../Datasets/Noisy pipes/Subtractive/Elegra/MLonOrig/2mm btm a_speckled.png"
input_dir = "../Datasets/Noisy pipes/Subtractive/Elegra/MLonOrig"
output_dir = "../Datasets/Noisy pipes/Subtractive/Elegra/MLonOrig/cropped"
os.makedirs(output_dir, exist_ok=True)
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("Error: Could not open the image.")
else:
    # Display the image and set up the mouse callback
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_square)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # Press 'c' to confirm the crop area and process all images
        if key == ord("c"):
            if len(ref_point) == 2:
                crop_images(input_dir, output_dir, ref_point)
                break

        # Press 'q' to quit without cropping
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

# python
# import cv2
# import os
#
# ROOT = "../Datasets/Noisy pipes/Subtractive"
# VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".JPEG")
#
# ref_point = []
# cropping = False
# select_img = None
#
# def draw_square(event, x, y, flags, param):
#     global ref_point, cropping, select_img
#     if event == cv2.EVENT_LBUTTONDOWN:
#         ref_point = [(x, y)]
#         cropping = True
#     elif event == cv2.EVENT_MOUSEMOVE and cropping:
#         img_copy = select_img.copy()
#         side = max(abs(x - ref_point[0][0]), abs(y - ref_point[0][1]))
#         end = (ref_point[0][0] + side, ref_point[0][1] + side)
#         cv2.rectangle(img_copy, ref_point[0], end, (0, 255, 0), 2)
#         cv2.imshow("Select crop (L-click/drag). Press 'c' to confirm.", img_copy)
#     elif event == cv2.EVENT_LBUTTONUP:
#         cropping = False
#         side = max(abs(x - ref_point[0][0]), abs(y - ref_point[0][1]))
#         end = (ref_point[0][0] + side, ref_point[0][1] + side)
#         # clamp to image bounds
#         end = (min(end[0], select_img.shape[1]-1), min(end[1], select_img.shape[0]-1))
#         ref_point.append(end)
#         cv2.rectangle(select_img, ref_point[0], ref_point[1], (0,255,0), 2)
#         cv2.imshow("Select crop (L-click/drag). Press 'c' to confirm.", select_img)
#
# def find_first_image():
#     folders = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and d.startswith("Capasee2002_Noise")])
#     if not folders:
#         raise FileNotFoundError(f"No `Capasee2002_Noise*` folders found under `{ROOT}`")
#     for f in folders:
#         folder_path = os.path.join(ROOT, f)
#         for name in sorted(os.listdir(folder_path)):
#             if name.endswith(VALID_EXTS):
#                 return os.path.join(folder_path, name)
#     raise FileNotFoundError("No images found in the first noise folder.")
#
# def crop_and_save_all(selection):
#     sel_h, sel_w = selection.shape[:2]
#     sx, sy = ref_point[0]
#     ex, ey = ref_point[1]
#     sel_side = ex - sx
#
#     folders = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and d.startswith("Capasee2002_Noise")])
#     for folder in folders:
#         folder_path = os.path.join(ROOT, folder)
#         out_dir = os.path.join(folder_path, "cropped")
#         os.makedirs(out_dir, exist_ok=True)
#
#         for fname in sorted(os.listdir(folder_path)):
#             if not fname.endswith(VALID_EXTS):
#                 continue
#             src_path = os.path.join(folder_path, fname)
#             img = cv2.imread(src_path)
#             if img is None:
#                 print(f"Skipped (can't read): {src_path}")
#                 continue
#             h, w = img.shape[:2]
#             # scale selection to current image size
#             scale_x = w / sel_w
#             scale_y = h / sel_h
#             # use min scale to keep square and avoid going out of bounds
#             scale = min(scale_x, scale_y)
#             start_x = int(round(sx * scale))
#             start_y = int(round(sy * scale))
#             side = max(1, int(round(sel_side * scale)))
#             end_x = start_x + side
#             end_y = start_y + side
#             # clamp to image bounds
#             if end_x > w:
#                 end_x = w
#                 start_x = max(0, end_x - side)
#             if end_y > h:
#                 end_y = h
#                 start_y = max(0, end_y - side)
#             cropped = img[start_y:end_y, start_x:end_x]
#             if cropped.size == 0:
#                 print(f"Skipped (empty crop) for {src_path}")
#                 continue
#             save_path = os.path.join(out_dir, fname)
#             cv2.imwrite(save_path, cropped)
#         print(f"Cropped images saved in: {out_dir}")
#
# if __name__ == "__main__":
#     try:
#         first_image_path = find_first_image()
#     except FileNotFoundError as e:
#         print(e)
#         raise SystemExit(1)
#
#     select_img = cv2.imread(first_image_path)
#     if select_img is None:
#         print(f"Could not open `{first_image_path}`")
#         raise SystemExit(1)
#
#     cv2.namedWindow("Select crop (L-click/drag). Press 'c' to confirm.", cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback("Select crop (L-click/drag). Press 'c' to confirm.", draw_square)
#     cv2.imshow("Select crop (L-click/drag). Press 'c' to confirm.", select_img)
#
#     while True:
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("c"):
#             if len(ref_point) == 2:
#                 cv2.destroyAllWindows()
#                 crop_and_save_all(select_img)
#                 break
#             else:
#                 print("Please select a square crop first (drag with left mouse button).")
#         elif key == ord("q") or key == 27:
#             cv2.destroyAllWindows()
#             print("Cancelled by user.")
#             break
