import os
import cv2
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from PIL import Image, ImageTk
import json


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


checkbox_style = """
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px black;
    border-radius: 5px;
    background: none;   /* unchecked background */
}

QCheckBox::indicator:checked {
    background: rgb(74, 118, 189);   /* checked background */
    border: 1px black;
    border: 1px solid black;
}

QCheckBox::indicator:unchecked {
    background: none;   /* unchecked background */
    border: 1px black;
    border: 1px solid black;
}
"""

button_style = """
QPushButton {
    border: 1px solid gray;
    border-radius: 5px;
    padding: 3px;
}
QPushButton:enabled {
    background-color: rgb(197, 228, 255);
    border: 1px solid gray;
}
QPushButton:disabled {
    background-color: rgba(255, 255, 255, 127);
    color: gray;
    border: 1px solid lightgray;
}
"""

progress_style = """
QProgressBar {
    text-align: center;
    border-radius: 5px;
    border: 1px solid rgb(74, 118, 189);
}
QProgressBar:enabled {
    text-align: center;
    border-radius: 5px;
    border: 1px solid rgb(74, 118, 189);
}
QProgressBar:disabled {
    border: 1px solid lightgray;
    color: lightgray;
}
QProgressBar::chunk {
    background-color: rgba(74, 118, 189, 127);
    width: 10px;
    margin: 0.5px;
}
"""

textbox_style = """
    QPlainTextEdit {
        border: 1px solid gray;
        border-radius: 5px;
        padding: 3px;
    }
"""

QWidget_style = """
    QWidget {
        border: 3px solid rgb(197, 228, 255);
        border-radius: 5px;
        padding: 3px;
"""

def calculate_pipe_length_for_group(data_dir, diameter, folders):
    top_path = os.path.join(data_dir, folders["top"], "special_distances_unetXresnet18.json")
    bot_path = os.path.join(data_dir, folders["bot"], "special_distances_unetXresnet18.json")

    pipe_lengths = []

    if os.path.exists(top_path) and os.path.exists(bot_path):
        with open(top_path, 'r') as top_file:
            top_data = json.load(top_file)
            min_top_distance_mm = top_data.get("MinTopDistance_mm")

        with open(bot_path, 'r') as bot_file:
            bot_data = json.load(bot_file)
            max_bottom_distance_mm = bot_data.get("MaxBottomDistance_mm")

        if min_top_distance_mm is not None and max_bottom_distance_mm is not None:
            pipe_length_mm = max_bottom_distance_mm - min_top_distance_mm
            pipe_lengths.append((diameter, pipe_length_mm))

    return pipe_lengths


def calculate_pipe_lengths(data_dir, batch_size=5):
    diameter_groups = {}
    video_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # Group folders by diameter
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

    # Split the diameter_groups into smaller batches
    diameter_group_items = list(diameter_groups.items())
    num_batches = (len(diameter_group_items) // batch_size) + (1 if len(diameter_group_items) % batch_size else 0)

    # Process the batches in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(num_batches):
            batch = diameter_group_items[i * batch_size : (i + 1) * batch_size]
            futures.append(executor.submit(process_batch, data_dir, batch))

        # Collect results from all batches
        for future in as_completed(futures):
            pipe_lengths.extend(future.result())

    return pipe_lengths


def process_batch(data_dir, batch):
    pipe_lengths = []
    for diameter, folders in batch:
        if folders["top"] and folders["bot"]:
            pipe_lengths.extend(calculate_pipe_length_for_group(data_dir, diameter, folders))
    return pipe_lengths


def save_image(image_tensor, save_path):
    image = Image.fromarray(image_tensor.numpy())
    image.save(save_path)


def save_images_parallel(image_tensors, save_paths):
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on system I/O
        executor.map(save_image, image_tensors, save_paths)


def process_batch_with_args(args):
    """
    Wrapper to unpack arguments and call `process_batch`.
    """
    batch_paths, output_dir, model, device, transform = args
    return process_batch(batch_paths, output_dir, model, device, transform)


# Move this function to the top level of your script
def process_batch(batch_paths, output_dir, model, device, transform):
    """
    Processes a batch of images, performs segmentation, and saves the masks.
    """
    results = []
    for img_path in batch_paths:
        # Load and preprocess image
        with Image.open(img_path) as img:
            img = img.convert('L')  # Convert to grayscale (1 channel)
            img_tensor = transform(img).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).cpu().numpy().squeeze().astype(np.uint8) * 255

        # Postprocess mask and save
        processed_mask = prediction
        save_path = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '_mask.png'))
        Image.fromarray(processed_mask).save(save_path)
        results.append(save_path)

    return results


def process_masks_parallel(mask_files, masks_dir, video_folder):
    batch_size = 10
    batches = [mask_files[i:i + batch_size] for i in range(0, len(mask_files), batch_size)]
    args = [(batch, masks_dir, video_folder) for batch in batches]

    # Shared variables for aggregation
    max_bottom_distance = float('-inf')
    min_top_distance = float('inf')
    all_results = []

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_batch, args)

        for batch_results, batch_max_bottom, batch_min_top in results:
            all_results.extend(batch_results)

            # Aggregate max/min distances across batches
            max_bottom_distance = max(max_bottom_distance, batch_max_bottom)
            min_top_distance = min(min_top_distance, batch_min_top)

    return all_results, max_bottom_distance, min_top_distance


def segment_images_batch_parallel(input_dir, output_dir, model, device, transform, batch_size=10):
    """
    Segments images in parallel using batches.
    """
    # Prepare paths
    os.makedirs(output_dir, exist_ok=True)
    image_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")]

    # Split image paths into batches
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    # Prepare arguments as tuples for each batch
    args = [(batch, output_dir, model, device, transform) for batch in batches]

    # Process in parallel
    with ProcessPoolExecutor() as executor:
        all_mask_files = sum(executor.map(process_batch_with_args, args), [])

    return all_mask_files


def extract_top_bottom_positions(mask):
    non_zero_rows = np.where(mask > 0)[0]
    if len(non_zero_rows) > 0:
        top_position = non_zero_rows[0]
        bottom_position = non_zero_rows[-1]
        return top_position, bottom_position
    else:
        return None, None  # No mask found

def processing_phase(data_dir, cropping_coordinates, conversion_factors, model, device, transform):
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    import os
    import json

    def process_video_folder(video_folder):
        video_folder_path = os.path.join(data_dir, video_folder)
        frames_dir = os.path.join(video_folder_path, "frames")
        cropped_dir = os.path.join(video_folder_path, "cropped")
        masks_dir = os.path.join(video_folder_path, "masks")

        # Check if cropping coordinates exist for this video folder
        if video_folder not in cropping_coordinates:
            print(f"Skipping {video_folder}: No cropping coordinates available.")
            return

        # Step 1: Crop images and save cropped versions
        parallel_crop_images(frames_dir, cropped_dir, cropping_coordinates[video_folder])

        # Step 2: Segment images and save masks
        segment_images_batch_parallel(cropped_dir, masks_dir, model, device, transform, batch_size=10)

        # Step 3: Process masks in parallel with threshold logic
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

        distances, max_bottom_distance, min_top_distance = process_masks_parallel(
            mask_files, masks_dir, video_folder
        )

        # Step 4: Convert distances using the appropriate conversion factor
        conversion_factor = conversion_factors[video_folder]
        distances = convert_distances(distances, conversion_factor)

        # Step 5: Finalize and save distances
        for distance in distances:
            distance["DistanceFromTop"] = int(distance["DistanceFromTop"])
            distance["DistanceFromBottom"] = int(distance["DistanceFromBottom"])
            distance["DistanceFromTop_cm"] = float(distance["DistanceFromTop_cm"])
            distance["DistanceFromBottom_cm"] = float(distance["DistanceFromBottom_cm"])

        distances_path = os.path.join(video_folder_path, 'distances.json')
        with open(distances_path, 'w') as f:
            json.dump(distances, f, indent=4)

        # Step 6: Compute special distances and save
        max_bottom_distance_cm = max_bottom_distance * conversion_factor if max_bottom_distance > float(
            '-inf') else None
        min_top_distance_cm = min_top_distance * conversion_factor if min_top_distance < float('inf') else None

        special_distances = {
            "MaxBottomDistance_pixels": int(max_bottom_distance) if max_bottom_distance_cm is not None else None,
            "MaxBottomDistance_cm": max_bottom_distance_cm,
            "MinTopDistance_pixels": int(min_top_distance) if min_top_distance_cm is not None else None,
            "MinTopDistance_cm": min_top_distance_cm
        }

        special_distances_path = os.path.join(video_folder_path, 'special_distances_unet.json')
        with open(special_distances_path, 'w') as f:
            json.dump(special_distances, f, indent=4)

    # Process video folders in parallel
    video_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust `max_workers` based on system resources
        list(tqdm(executor.map(process_video_folder, video_folders), total=len(video_folders), desc="Processing videos"))