import os
from PIL import Image

input_folder = "../Datasets/Elegra"  # Replace with your folder path
output_folder = "../Datasets/Elegra_JPG"  # Replace with your desired output folder
os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

for filename in os.listdir(input_folder):
    if filename.endswith(".bmp"):
        bmp_path = os.path.join(input_folder, filename)
        jpg_path = os.path.join(output_folder, filename.replace(".bmp", ".jpg"))

        image = Image.open(bmp_path)
        image.convert("RGB").save(jpg_path, "JPEG", quality=95)

        print(f"Converted {filename} to {jpg_path}")
