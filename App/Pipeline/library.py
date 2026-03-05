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


def extract_top_bottom_positions(mask):
    non_zero_rows = np.where(mask > 0)[0]
    if len(non_zero_rows) > 0:
        top_position = non_zero_rows[0]
        bottom_position = non_zero_rows[-1]
        return top_position, bottom_position
    else:
        return None, None  # No mask found