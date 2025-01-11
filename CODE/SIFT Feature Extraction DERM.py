import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Paths to DERM dataset folders
benign_path = r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\Dataset-6k\Benign Cropped"
malignant_path = r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\Dataset-6k\Malignant Cropped"
output_path = r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\Dataset-6k"
visualization_path = os.path.join(output_path, r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\New_Keypoint_Visualizations")
os.makedirs(visualization_path, exist_ok=True)  # Create directory for keypoint visualizations

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Resize dimensions
resize_dim = (512, 512)

def load_and_resize_images(folder, size):
    """Load all images from a folder, convert to grayscale, and resize."""
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, size)
            images.append(resized_img)
            filenames.append(filename)
    return images, filenames

def extract_and_visualize_sift_features(images, filenames, class_label):
    """Extract SIFT features and visualize keypoints on images."""
    descriptors_all = []
    for img, filename in zip(images, filenames):
        # Detect SIFT keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:  # Skip images with no descriptors
            descriptors_all.append(descriptors)
        
        # Visualize keypoints
        image_with_keypoints = cv2.drawKeypoints(
            img, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Save visualization
        visualization_file = os.path.join(
            visualization_path, f"{class_label}_{filename}"
        )
        cv2.imwrite(visualization_file, image_with_keypoints)
    return descriptors_all

# Load and resize benign and malignant images
benign_images, benign_filenames = load_and_resize_images(benign_path, resize_dim)
malignant_images, malignant_filenames = load_and_resize_images(malignant_path, resize_dim)

# Extract SIFT features and visualize keypoints
print("Processing Benign Images...")
benign_descriptors = extract_and_visualize_sift_features(benign_images, benign_filenames, "Benign")

print("Processing Malignant Images...")
malignant_descriptors = extract_and_visualize_sift_features(malignant_images, malignant_filenames, "Malignant")

# Combine descriptors and create labels (0 for Benign, 1 for Malignant)
descriptors_all = benign_descriptors + malignant_descriptors
labels = [0] * len(benign_descriptors) + [1] * len(malignant_descriptors)

# Flatten the descriptors and labels for saving
flattened_descriptors = [desc for descriptors in descriptors_all for desc in descriptors]
flattened_labels = np.repeat(labels, [len(desc) for desc in descriptors_all])

# Convert to numpy arrays for saving
flattened_descriptors = np.array(flattened_descriptors)
flattened_labels = np.array(flattened_labels)

# Save features and labels as a compressed file
output_file = os.path.join(output_path, 'new_sift_features.npz')
np.savez_compressed(output_file, features=flattened_descriptors, labels=flattened_labels)

print(f"SIFT features and labels saved to {output_file}")
print(f"Keypoint visualizations saved in {visualization_path}")
