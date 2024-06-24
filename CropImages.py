import os
import cv2
import numpy as np
from tkinter import Tk, filedialog, simpledialog
import shutil


def select_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path


def select_directory():
    root = Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory()
    root.destroy()
    return directory_path


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


def crop_image(img, ref_point):
    x_start = ref_point[0][0]
    y_start = ref_point[0][1]
    side_length = ref_point[1][0] - ref_point[0][0]

    cropped_img = img[y_start:y_start + side_length, x_start:x_start + side_length]
    return cropped_img


if __name__ == "__main__":
    # Select the initial image
    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting.")
        exit()

    # Load the image
    image = cv2.imread(file_path)
    clone = image.copy()

    # Set up mouse callback to draw square
    ref_point = []
    cropping = False
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_square)

    # Display the image and wait for user to draw the square
    print("Draw a square on the image and press 'c' to crop.")
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            break
        elif key == ord("r"):
            image = clone.copy()

    cv2.destroyAllWindows()

    if len(ref_point) == 2:
        cropped_img = crop_image(clone, ref_point)
        cv2.imshow("Cropped Image", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Select the directory with images to crop
        directory_path = select_directory()
        if not directory_path:
            print("No directory selected. Exiting.")
            exit()

        # Create a new folder to save cropped images
        save_directory = os.path.join(directory_path, "cropped_images")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Loop through the directory and crop all images
        for filename in os.listdir(directory_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(directory_path, filename)
                img = cv2.imread(img_path)

                cropped_img = crop_image(img, ref_point)
                save_path = os.path.join(save_directory, filename)
                cv2.imwrite(save_path, cropped_img)

        print(f"Cropped images saved in: {save_directory}")
    else:
        print("Cropping cancelled.")
