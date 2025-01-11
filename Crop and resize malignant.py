
from PIL import Image
import os

# Input and output directories
input_dir = r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\Dataset-6k\Malignant"  # Replace with the folder containing your images
output_dir = r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\Dataset-6k\Malignant Cropped"  # Replace with the folder to save processed images
os.makedirs(output_dir, exist_ok=True)

# Crop percentage (e.g., 70% of the smallest dimension)
crop_percentage = 0.6
output_size = (512, 512)  # Final resize dimensions

# Process each image
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Include supported formats
        image_path = os.path.join(input_dir, filename)
        with Image.open(image_path) as img:
            width, height = img.size
            # Calculate crop dimensions
            crop_size = int(min(width, height) * crop_percentage)
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            right = (width + crop_size) / 2
            bottom = (height + crop_size) / 2
            # Center crop
            cropped_img = img.crop((left, top, right, bottom))
            # Resize to the target size
            resized_img = cropped_img.resize(output_size)
            # Save the processed image
            resized_img.save(os.path.join(output_dir, filename))

print("Images cropped and resized successfully!")
