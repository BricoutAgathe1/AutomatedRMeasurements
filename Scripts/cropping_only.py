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
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip if the image is not loaded properly
            cropped_img = crop_image(img, ref_point)
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
image_path = "../Datasets/Noisy pipes/Noise=0.6/2mm btm a_speckled.png"
input_dir = "../Datasets/Noisy pipes/Noise=0.6"
output_dir = "../Datasets/Noisy pipes/Noise=0.6/cropped"
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
