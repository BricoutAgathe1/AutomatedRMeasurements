import json
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Path to the VIA annotations file
annotations_file = '../Annotations/Chris_scanners/SamsungR20_LA2-14A_json.json'

# Directory to save mask images
masks_dir = '../Datasets/Chris_scanners/Samsung R20/LA2-14A'
os.makedirs(masks_dir, exist_ok=True)

# Load annotations
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Debugging: Print the first few annotations
print(json.dumps(list(annotations.items())[:5], indent=2))

# Iterate through the annotations and create masks
for key, file_annotations in annotations.items():
    filename = file_annotations['filename']
    width = file_annotations.get('width', 0)  # Default to 0 if width is not present
    height = file_annotations.get('height', 0)  # Default to 0 if height is not present
    regions = file_annotations.get('regions', {})

    if width == 0 or height == 0:
        # Load image to get its dimensions if not available in annotations
        image_path = os.path.join('../Datasets/Chris_scanners/Samsung R20/LA2-14A', filename)
        with Image.open(image_path) as img:
            width, height = img.size

    print(f"Processing {filename} with dimensions ({width}, {height})")

    # Create an empty mask image
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for region in regions:
        print(type(region), region)  # Debugging: Check type and content of region
        shape_attributes = region['shape_attributes']
        if shape_attributes['name'] == 'polygon':
            points = list(zip(shape_attributes['all_points_x'], shape_attributes['all_points_y']))
            print(f"Drawing polygon with points: {points}")
            draw.polygon(points, outline=255, fill=255)  # Use 255 for white mask regions
        else:
            print(f"Skipping non-polygon shape: {shape_attributes['name']}")

    # Convert mask to binary (0 and 255) and check values
    mask_array = mask.load()
    for y in range(height):
        for x in range(width):
            if mask_array[x, y] != 0:
                mask_array[x, y] = 255

    # Save the mask image
    mask_path = os.path.join(masks_dir, f"{os.path.splitext(filename)[0]}_mask.png")
    mask.save(mask_path)

print("Masks generated successfully.")

# Visualize the original image and its mask
for key, file_annotations in annotations.items():
    filename = file_annotations['filename']
    regions = file_annotations.get('regions', {})

    if not regions:
        continue

    image_path = os.path.join('../Datasets/Chris_scanners/Samsung R20/LA2-14A', filename)
    mask_path = os.path.join(masks_dir, f"{os.path.splitext(filename)[0]}_mask.png")

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)

    # Display the image and its mask
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Generated Mask")
    plt.show()

    # Break after one example for inspection
    break
