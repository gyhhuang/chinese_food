import os
from PIL import Image
import pillow_avif

input_dir = "test_data"
output_dir = "real_world_set2"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Convert all AVIF images
for filename in os.listdir(input_dir):
    if filename.endswith(".avif"):
        avif_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".avif", ".jpg"))

        image = Image.open(avif_path)
        image.save(output_path)
        print(f"Converted: {filename} -> {output_path}")
