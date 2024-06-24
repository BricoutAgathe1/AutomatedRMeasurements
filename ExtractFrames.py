import cv2
import os
import shutil
import random

# Function to create directories if they don't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Define directories
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

create_dir(train_dir)
create_dir(val_dir)
create_dir(test_dir)

# Read the video from specified path
video_path = 'Lq e9 9L/3mm bot.mp4'
cam = cv2.VideoCapture(video_path)

# Extract frames from the video
frames = []
currentframe = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frames.append(frame)
    currentframe += 1

cam.release()
cv2.destroyAllWindows()

# Shuffle frames before splitting
random.shuffle(frames)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Calculate split indices
total_frames = len(frames)
train_idx = int(train_ratio * total_frames)
val_idx = int((train_ratio + val_ratio) * total_frames)

# Split frames
train_frames = frames[:train_idx]
val_frames = frames[train_idx:val_idx]
test_frames = frames[val_idx:]

# Save frames to respective directories
def save_frames(frames, directory):
    for idx, frame in enumerate(frames):
        frame_name = os.path.join(directory, f'frame{idx}.jpg')
        cv2.imwrite(frame_name, frame)

save_frames(train_frames, train_dir)
save_frames(val_frames, val_dir)
save_frames(test_frames, test_dir)

print(f"Total frames: {total_frames}")
print(f"Training frames: {len(train_frames)}")
print(f"Validation frames: {len(val_frames)}")
print(f"Testing frames: {len(test_frames)}")

print("Frames extraction and splitting complete.")
