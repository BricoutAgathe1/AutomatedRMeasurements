import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from multiprocessing import Pool
from PIL import Image, ImageTk
from tkinter import Tk, Canvas, Entry, Label, Button, filedialog
import json
from tqdm import tqdm
import time

start_time = time.time()


# UNet Model Definition
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoding path
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)

        # Decoding path with upsampling
        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)  # Match channels after concatenation
        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)  # Match channels after concatenation
        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)  # Match channels after concatenation
        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)  # Match channels after concatenation

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.functional.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(nn.functional.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(nn.functional.max_pool2d(enc3, kernel_size=2))
        bottleneck = self.bottleneck(nn.functional.max_pool2d(enc4, kernel_size=2))

        # Decoding path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)


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
def crop_image(img, ref_point, target_size=(256, 256)):
    x_start = ref_point[0][0]
    y_start = ref_point[0][1]
    side_length = ref_point[1][0] - ref_point[0][0]
    cropped_img = img[y_start:y_start + side_length, x_start:x_start + side_length]

    # Resize the cropped image to 256x256
    resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img


def crop_and_save_image(args):
    img_path, output_dir, ref_point = args
    img = cv2.imread(img_path)
    if img is None:
        return  # Skip if the image is not loaded properly

    cropped_resized_img = crop_image(img, ref_point)
    save_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, cropped_resized_img)


# Function to crop and resize all images in a directory
def parallel_crop_images(input_dir, output_dir, ref_point):
    os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments for multiprocessing
    image_paths = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.endswith(".jpg") or filename.endswith(".png")
    ]
    args = [(img_path, output_dir, ref_point) for img_path in image_paths]

    # Use multiprocessing Pool
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(crop_and_save_image, args)
    # print(f"Cropped and resized images saved in: {output_dir}")


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


# Define ImageDataset at the top level
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # Open as grayscale
        if self.transform:
            img = self.transform(img)
        return img, img_path


def segment_images_batch(input_dir, output_dir, model, device, transform, batch_size=16):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    distances = []

    image_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")]
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    with torch.no_grad():
        for batch_imgs, batch_paths in dataloader:
            batch_imgs = batch_imgs.to(device)
            outputs = model(batch_imgs)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8) * 255

            for pseudolabel, img_path in zip(predictions, batch_paths):
                # Keep only the largest connected component
                processed_mask = keep_largest_connected_component(pseudolabel)

                pseudolabel_img = Image.fromarray(processed_mask)
                save_path = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '_mask.png'))
                pseudolabel_img.save(save_path)

                if np.any(processed_mask):  # Check if any segmentation occurred
                    top_pos, bottom_pos = extract_top_bottom_positions(processed_mask)
                    if top_pos is not None and bottom_pos is not None:
                        distances.append({
                            'MaskFile': save_path,
                            'DistanceFromTop': top_pos,
                            'DistanceFromBottom': bottom_pos
                        })
                    else:
                        distances.append({
                            'MaskFile': save_path,
                            'DistanceFromTop': -1,
                            'DistanceFromBottom': -1
                        })

    # Save distances to a file
    distances_serializable = []
    for entry in distances:
        distances_serializable.append({
            'MaskFile': entry['MaskFile'],
            'DistanceFromTop': int(entry['DistanceFromTop']),
            'DistanceFromBottom': int(entry['DistanceFromBottom'])
        })

    with open('../Distances/distances_unet.json', 'w') as f:
        json.dump(distances_serializable, f)

    # print(f"Distances successfully saved for video: {input_dir}")


def extract_top_bottom_positions(mask):
    non_zero_rows = np.where(mask > 0)[0]
    if len(non_zero_rows) > 0:
        top_position = non_zero_rows[0]
        bottom_position = non_zero_rows[-1]
        return top_position, bottom_position
    else:
        return None, None  # No mask found


def select_directory():
    root = Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory()
    root.destroy()
    return directory_path


class DistanceInputApp:
    def __init__(self, root, image):
        self.root = root
        self.image = image
        self.canvas = Canvas(root, width=image.width, height=image.height)
        self.canvas.pack()
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.image_tk)
        self.canvas.bind("<Button-1>", self.on_click)
        self.points = []
        self.distance_entry = Entry(root)
        self.distance_entry.pack()
        self.label = Label(root, text="Enter the real-world distance in cm for the drawn line:")
        self.label.pack()
        self.button = Button(root, text="Submit", command=self.on_submit)
        self.button.pack()
        self.real_world_distance = None

    def on_click(self, event):
        if len(self.points) < 2:
            x, y = event.x, event.y
            if len(self.points) == 1:
                # Ensure the second point has the same x-coordinate as the first one
                x = self.points[0][0]
            self.points.append((x, y))
            if len(self.points) == 2:
                self.draw_line()
                self.distance_entry.focus()

    def draw_line(self):
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        self.canvas.create_line(x1, y1, x2, y2, fill="red")

    def on_submit(self):
        try:
            self.real_world_distance = float(self.distance_entry.get())
            self.root.quit()
        except ValueError:
            print("Please enter a valid number.")


def get_distance_conversion_factor(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((256, 256), Image.BILINEAR)
    root = Tk()
    app = DistanceInputApp(root, image)
    root.mainloop()
    root.destroy()  # Make sure to destroy the Tkinter root window after use

    if len(app.points) == 2 and app.real_world_distance is not None:
        (x1, y1), (x2, y2) = app.points
        pixel_distance = abs(y2 - y1)  # Vertical distance only
        conversion_factor = app.real_world_distance / pixel_distance
        return conversion_factor
    else:
        print("Distance conversion factor calculation failed.")
        return None


def interactive_cropping_phase(data_dir):
    cropping_coordinates = {}
    conversion_factors = {}

    for video_folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, video_folder)):
            base_name = video_folder
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
            print(f"Draw a square on the image for video: {video_folder} and press 'c' to crop, 'r' to redo.")
            while True:
                cv2.imshow("image", image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    break
                elif key == ord("r"):
                    image = cv2.imread(first_image_path)
                    ref_point = []

            cv2.destroyAllWindows()

            cropping_coordinates[video_folder] = ref_point

            # Ensure that the cropped directory exists and save the cropped image there
            os.makedirs(cropped_dir, exist_ok=True)
            if ref_point:
                x1, y1 = ref_point[0]
                x2, y2 = ref_point[1]
                cropped_img = image[y1:y2, x1:x2]
                cropped_img_path = os.path.join(cropped_dir, 'cropped_first_image.jpg')
                cv2.imwrite(cropped_img_path, cropped_img)

            # Get the conversion factor
            first_crop_path = os.path.join(cropped_dir, 'cropped_first_image.jpg')
            conversion_factor = get_distance_conversion_factor(first_crop_path)
            conversion_factors[video_folder] = conversion_factor

    return cropping_coordinates, conversion_factors


def convert_distances(distances, conversion_factor):
    for distance in distances:
        distance["DistanceFromTop_cm"] = distance["DistanceFromTop"] * conversion_factor
        distance["DistanceFromBottom_cm"] = distance["DistanceFromBottom"] * conversion_factor
    return distances


def keep_largest_connected_component(mask):
    """
    Retain only the largest connected component in the binary mask.
    Args:
        mask (numpy.ndarray): Binary mask of the segmentation (values: 0 or 255).

    Returns:
        numpy.ndarray: Processed mask with only the largest connected component.
    """
    # Find all connected components (white regions in the binary mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # If there's only one label (background), return the mask as is
    if num_labels <= 1:
        return mask

    # Identify the largest component (excluding the background, label 0)
    largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a new binary mask for the largest component
    largest_component_mask = np.zeros_like(mask)
    largest_component_mask[labels == largest_component_idx] = 255

    return largest_component_mask


def processing_phase(data_dir, cropping_coordinates, conversion_factors, model, device, transform, threshold=0.05):
    video_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # Use tqdm to display progress for video folder processing
    for video_folder in tqdm(video_folders, desc="Processing video folders"):
        video_folder_path = os.path.join(data_dir, video_folder)
        frames_dir = os.path.join(video_folder_path, "frames")
        cropped_dir = os.path.join(video_folder_path, "cropped")
        masks_dir = os.path.join(video_folder_path, "masks")

        if video_folder in cropping_coordinates:
            parallel_crop_images(frames_dir, cropped_dir, cropping_coordinates[video_folder])
            segment_images_batch(cropped_dir, masks_dir, model, device, transform, batch_size=16)

            distances = []
            max_bottom_distance = float('-inf')
            min_top_distance = float('inf')

            mask_files = os.listdir(masks_dir)
            # Use tqdm for progress within each video folder's mask processing
            for filename in tqdm(mask_files, desc=f"Processing masks in {video_folder}", leave=False):
                mask_path = os.path.join(masks_dir, filename)
                mask = Image.open(mask_path)
                mask = np.array(mask)
                top_position, bottom_position = extract_top_bottom_positions(mask)

                if top_position is not None and bottom_position is not None:
                    distances.append({
                        "Frame": filename,
                        "DistanceFromTop": int(top_position),
                        "DistanceFromBottom": int(bottom_position)
                    })

                    if "bot" in video_folder.lower():
                        if max_bottom_distance == float('-inf'):
                            max_bottom_distance = bottom_position
                        elif abs(bottom_position - max_bottom_distance) <= threshold * max_bottom_distance:
                            max_bottom_distance = max(max_bottom_distance, bottom_position)

                    if "top" in video_folder.lower():
                        if min_top_distance == float('inf'):
                            min_top_distance = top_position
                        elif abs(top_position - min_top_distance) <= threshold * min_top_distance:
                            min_top_distance = min(min_top_distance, top_position)

                else:
                    distances.append({
                        "Frame": filename,
                        "DistanceFromTop": -1,
                        "DistanceFromBottom": -1
                    })

            conversion_factor = conversion_factors[video_folder]
            distances = convert_distances(distances, conversion_factor)

            for distance in distances:
                distance["DistanceFromTop"] = int(distance["DistanceFromTop"])
                distance["DistanceFromBottom"] = int(distance["DistanceFromBottom"])
                distance["DistanceFromTop_cm"] = float(distance["DistanceFromTop_cm"])
                distance["DistanceFromBottom_cm"] = float(distance["DistanceFromBottom_cm"])

            with open(os.path.join(video_folder_path, 'distances.json'), 'w') as f:
                json.dump(distances, f, indent=4)

            if max_bottom_distance > float('-inf'):
                max_bottom_distance_cm = max_bottom_distance * conversion_factor
            else:
                max_bottom_distance_cm = None

            if min_top_distance < float('inf'):
                min_top_distance_cm = min_top_distance * conversion_factor
            else:
                min_top_distance_cm = None

            special_distances = {
                "MaxBottomDistance_pixels": int(max_bottom_distance) if max_bottom_distance_cm is not None else None,
                "MaxBottomDistance_cm": max_bottom_distance_cm,
                "MinTopDistance_pixels": int(min_top_distance) if min_top_distance_cm is not None else None,
                "MinTopDistance_cm": min_top_distance_cm
            }

            with open(os.path.join(video_folder_path, 'special_distances_unet.json'), 'w') as f:
                json.dump(special_distances, f, indent=4)


def calculate_pipe_lengths(data_dir):
    diameter_groups = {}
    video_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    for video_folder in video_folders:
        diameter = video_folder.split()[0][:-2]
        diameter = float(diameter)

        if diameter not in diameter_groups:
            diameter_groups[diameter] = {"top": None, "bot": None}

        if "top" in video_folder.lower():
            diameter_groups[diameter]["top"] = video_folder
        elif "bot" in video_folder.lower():
            diameter_groups[diameter]["bot"] = video_folder

    pipe_lengths = []

    # Use tqdm for processing each diameter group
    for diameter, folders in tqdm(diameter_groups.items(), desc="Calculating pipe lengths"):
        if folders["top"] and folders["bot"]:
            top_path = os.path.join(data_dir, folders["top"], "special_distances_unet.json")
            bot_path = os.path.join(data_dir, folders["bot"], "special_distances_unet.json")

            if os.path.exists(top_path) and os.path.exists(bot_path):
                with open(top_path, 'r') as top_file:
                    top_data = json.load(top_file)
                    min_top_distance_cm = top_data.get("MinTopDistance_cm")

                with open(bot_path, 'r') as bot_file:
                    bot_data = json.load(bot_file)
                    max_bottom_distance_cm = bot_data.get("MaxBottomDistance_cm")

                if min_top_distance_cm is not None and max_bottom_distance_cm is not None:
                    pipe_length_cm = max_bottom_distance_cm - min_top_distance_cm
                    pipe_lengths.append((diameter, pipe_length_cm))

                    print(f"Pipe length = {pipe_length_cm:.2f} cm for pipe of diameter {diameter:.1f} mm")

    return pipe_lengths


def main():

    data_dir = select_directory()

    # Load model and weights from initial training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load('../Model weights/best_segmentation_model_weights_unet.pth', map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Step 1: Extract frames from all videos in the selected directory
    video_files = [f for f in os.listdir(data_dir) if f.endswith(".mp4")]  # Assuming video files are in .mp4 format
    video_paths = [os.path.join(data_dir, video_file) for video_file in video_files]
    frames_dir = [os.path.join(data_dir, os.path.splitext(video_file)[0], "frames") for video_file in video_files]

    cropped_dir = [os.path.join(data_dir, os.path.splitext(video_file)[0], "cropped") for video_file in video_files]
    masks_dir = [os.path.join(data_dir, os.path.splitext(video_file)[0], "masks") for video_file in video_files]

    # Multiprocessing: Process all videos in parallel
    print("Extracting frames from videos...")
    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.starmap(extract_frames, zip(video_paths, frames_dir)), total=len(video_files)))

    print("Frame extraction completed in %s seconds." %(time.time()-start_time))

    # Step 2: Interactive cropping & distance conversion extraction
    cropping_coordinates, conversion_factors = interactive_cropping_phase(data_dir)

    # Step 3: Processing phase
    start_time_process = time.time()
    processing_phase(data_dir, cropping_coordinates, conversion_factors, model, device, transform, threshold=0.1)
    print("Images processed in %s seconds." %(time.time()-start_time_process))

    # Step 4: Calculate and print pipe lengths
    pipe_lengths = calculate_pipe_lengths(data_dir)



if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method('spawn')
    main()

    print("--- Testing complete: %s seconds ---" % (time.time() - start_time))
