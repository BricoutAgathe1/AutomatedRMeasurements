import cv2
import os
import random
import shutil

# Function to create directories if they don't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to extract frames from a video and save them to a specified directory
def extract_frames(video_path, output_dir, start_index):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = start_index

    while success:
        frame_name = os.path.join(output_dir, f"frame{count}.jpg")
        cv2.imwrite(frame_name, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    return count

# Function to shuffle frames and split them into train, val, and test sets
def shuffle_and_split(frames_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    # Get list of all frame files
    frame_files = os.listdir(frames_dir)
    total_frames = len(frame_files)

    # Shuffle the frame files
    random.shuffle(frame_files)

    # Calculate split indices
    train_idx = int(train_ratio * total_frames)
    val_idx = int((train_ratio + val_ratio) * total_frames)

    # Split the frame files
    train_frames = frame_files[:train_idx]
    val_frames = frame_files[train_idx:val_idx]
    test_frames = frame_files[val_idx:]

    # Copy frames to respective directories
    def copy_frames(frames, dest_dir):
        for frame in frames:
            src_path = os.path.join(frames_dir, frame)
            dest_path = os.path.join(dest_dir, frame)
            shutil.copy(src_path, dest_path)

    copy_frames(train_frames, train_dir)
    copy_frames(val_frames, val_dir)
    copy_frames(test_frames, test_dir)

    print(f"Total frames: {total_frames}")
    print(f"Training frames: {len(train_frames)}")
    print(f"Validation frames: {len(val_frames)}")
    print(f"Testing frames: {len(test_frames)}")

# Main function to process all videos in a directory
def process_videos(videos_dir, base_dir):
    # Create directories for frames and splits
    frames_dir = os.path.join(base_dir, 'frames')
    create_dir(frames_dir)

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    create_dir(train_dir)
    create_dir(val_dir)
    create_dir(test_dir)

    # Initialize frame counter
    frame_counter = 0

    # Process each video file in the directory
    for video_file in os.listdir(videos_dir):
        if video_file.endswith(".mp4") or video_file.endswith(".avi"):  # Add more video formats as needed
            video_path = os.path.join(videos_dir, video_file)
            print(f"Extracting frames from: {video_file}")
            frame_counter = extract_frames(video_path, frames_dir, frame_counter)

    # Shuffle and split the frames
    shuffle_and_split(frames_dir, train_dir, val_dir, test_dir)

# Example usage:
videos_dir = "Lq E9 9L - cropped"  # Directory containing the 14 videos
base_dir = 'Global_Dataset'  # Base directory for output
process_videos(videos_dir, base_dir)

print("Frames extraction and splitting complete.")
