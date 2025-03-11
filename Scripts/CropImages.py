import os
import cv2
import numpy as np
from tkinter import Tk, filedialog, simpledialog

# Function to extract frames from a video
def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames from {video_path}")

# Function to crop images
def crop_image(img, ref_point):
    x_start = ref_point[0][0]
    y_start = ref_point[0][1]
    side_length = ref_point[1][0] - ref_point[0][0]

    cropped_img = img[y_start:y_start + side_length, x_start:x_start + side_length]
    return cropped_img

def crop_images(input_dir, output_dir, ref_point):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip if the image is not loaded properly
            cropped_img = crop_image(img, ref_point)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, cropped_img)
    print(f"Cropped images saved in: {output_dir}")

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

def select_directory():
    root = Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory()
    root.destroy()
    return directory_path


def process_videos(data_dir):
    for video_filename in os.listdir(data_dir):
        if video_filename.endswith(".mp4"):  # Assuming video files are .mp4
            video_path = os.path.join(data_dir, video_filename)
            base_name = os.path.splitext(video_filename)[0]

            # Step 1: Extract frames
            frames_dir = os.path.join(data_dir, base_name, "frames")
            extract_frames(video_path, frames_dir)

    # Get cropping reference points for each video dataset
    for video_filename in os.listdir(data_dir):
        if video_filename.endswith(".mp4"):
            base_name = os.path.splitext(video_filename)[0]
            frames_dir = os.path.join(data_dir, base_name, "frames")
            cropped_dir = os.path.join(data_dir, base_name, "cropped")

            # Ask the user to crop the first image
            first_image_path = os.path.join(frames_dir, os.listdir(frames_dir)[0])
            global image, ref_point, cropping
            image = cv2.imread(first_image_path)
            ref_point = []
            cropping = False

            cv2.namedWindow("image")
            cv2.setMouseCallback("image", draw_square)

            # Display the image and wait for user to draw the square
            print(f"Draw a square on the image for video: {video_filename} and press 'c' to crop.")
            while True:
                cv2.imshow("image", image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    break
                elif key == ord("r"):
                    image = cv2.imread(first_image_path)
                    ref_point = []

            cv2.destroyAllWindows()

            if len(ref_point) == 2:
                # Step 2: Crop images based on user input
                crop_images(frames_dir, cropped_dir, ref_point)

                # # Step 3: Perform segmentation
                # segmented_dir = os.path.join(data_dir, base_name, "segmented")
                # segment_images(cropped_dir, segmented_dir)
            else:
                print(f"Cropping cancelled for video: {video_filename}")


if __name__ == "__main__":
    data_dir = select_directory()  # Path to your data directory
    if not data_dir:
        print("No directory selected. Exiting.")
        exit()
    process_videos(data_dir)