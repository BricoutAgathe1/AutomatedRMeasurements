import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageTk
from tkinter import Tk, Canvas, Entry, Label, Button, filedialog
import json
import time

start_time = time.time()


# Define the ResNet-based segmentation model
class ResNetSegmentation(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=2):
        super(ResNetSegmentation, self).__init__()
        if backbone == 'resnet18':
            self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            encoder_output_dim = 512
        elif backbone == 'resnet34':
            self.encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            encoder_output_dim = 512
        elif backbone == 'resnet50':
            self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            encoder_output_dim = 2048
        elif backbone == 'resnet101':
            self.encoder = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            encoder_output_dim = 2048
        elif backbone == 'resnet152':
            self.encoder = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            encoder_output_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  # Remove the fully connected layer
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_output_dim, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# frame extraction
# # Function to extract frames from a video
# def extract_frames(video_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")
#         cv2.imwrite(frame_filename, frame)
#         count += 1
#     cap.release()
#     print(f"Extracted {count} frames from {video_path}")


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

# Function to perform segmentation on images
def segment_images(input_dir, output_dir, model, device, transform):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    distances = []

    with torch.no_grad():
        for filename in os.listdir(input_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                output = model(img_tensor)
                pseudolabel = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8) * 255
                pseudolabel_img = Image.fromarray(pseudolabel)

                save_path = os.path.join(output_dir, filename.replace('.jpg', '_mask.png'))
                pseudolabel_img.save(save_path)

                if np.any(pseudolabel):  # Check if any segmentation occurred
                    top_pos, bottom_pos = extract_top_bottom_positions(pseudolabel)

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

    with open('distances.json', 'w') as f:
        json.dump(distances_serializable, f)

    print(f"Distances successfully saved for video: {input_dir}")

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
        self.label = Label(root, text="Enter the real-world distance in mm for the drawn line:")
        self.label.pack()
        self.button = Button(root, text="Submit", command=self.on_submit)
        self.button.pack()
        self.real_world_distance = None

    def on_click(self, event):
        if len(self.points) < 2:
            x, y = event.x, event.y
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
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)
    root = Tk()
    app = DistanceInputApp(root, image)
    root.mainloop()
    root.destroy()  # Make sure to destroy the Tkinter root window after use

    if len(app.points) == 2 and app.real_world_distance is not None:
        (x1, y1), (x2, y2) = app.points
        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
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
            print(f"Draw a square on the image for video: {video_folder} and press 'c' to crop.")
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
        distance["DistanceFromTop_mm"] = distance["DistanceFromTop"] * conversion_factor
        distance["DistanceFromBottom_mm"] = distance["DistanceFromBottom"] * conversion_factor
    return distances

def processing_phase(data_dir, cropping_coordinates, conversion_factors, model, device, transform):
    for video_folder in os.listdir(data_dir):
        video_folder_path = os.path.join(data_dir, video_folder)
        if os.path.isdir(video_folder_path):
            frames_dir = os.path.join(video_folder_path, "frames")
            cropped_dir = os.path.join(video_folder_path, "cropped")
            masks_dir = os.path.join(video_folder_path, "masks")

            if video_folder in cropping_coordinates:
                crop_images(frames_dir, cropped_dir, cropping_coordinates[video_folder])
                segment_images(cropped_dir, masks_dir, model, device, transform)

                distances = []

                for filename in os.listdir(masks_dir):
                    mask_path = os.path.join(masks_dir, filename)
                    mask = Image.open(mask_path)
                    mask = np.array(mask)
                    top_position, bottom_position = extract_top_bottom_positions(mask)

                    if top_position is not None and bottom_position is not None:
                        distances.append({
                            "Frame": filename,
                            "DistanceFromTop": int(top_position),   # Convert to native Python int
                            "DistanceFromBottom": int(bottom_position)  # Convert to native Python int
                        })
                    else:
                        distances.append({
                            "Frame": filename,
                            "DistanceFromTop": -1,
                            "DistanceFromBottom": -1
                        })

                conversion_factor = conversion_factors[video_folder]
                distances = convert_distances(distances, conversion_factor)

                # Ensure all values are JSON serializable
                for distance in distances:
                    distance["DistanceFromTop"] = int(distance["DistanceFromTop"])
                    distance["DistanceFromBottom"] = int(distance["DistanceFromBottom"])
                    distance["DistanceFromTop_mm"] = float(distance["DistanceFromTop_mm"])
                    distance["DistanceFromBottom_mm"] = float(distance["DistanceFromBottom_mm"])

                with open(os.path.join(video_folder_path, 'distances.json'), 'w') as f:
                    json.dump(distances, f, indent=4)


def main():
    data_dir = select_directory()

    model = ResNetSegmentation(backbone='resnet18', num_classes=2)  # Choose the appropriate backbone
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('best_combined_segmentation_model_weights.pth', map_location=device))
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    cropping_coordinates, conversion_factors = interactive_cropping_phase(data_dir)
    processing_phase(data_dir, cropping_coordinates, conversion_factors, model, device, transform)

if __name__ == "__main__":
    main()

print("--- Testing complete: %s seconds ---" % (time.time() - start_time))
