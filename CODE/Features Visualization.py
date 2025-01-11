import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# Define the save directory
save_path = r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\Dataset-6k"

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Load the SIFT features and labels from the saved .npz file
data = np.load(r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\Dataset-6k\new_sift_features.npz")
features = data['features']
labels = data['labels']

# Distribution Visualizations
def plot_label_distribution(labels, title):
    """Visualize the distribution of class labels."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=labels, palette='coolwarm')
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.xticks(ticks=[0, 1], labels=["Benign", "Malignant"])
    plt.show()

def plot_descriptor_distribution(features, labels):
    """Visualize the distribution of descriptors per image."""
    descriptors_per_image = [len(features[i:i + 100]) for i in range(0, len(features), 100)]
    plt.figure(figsize=(8, 5))
    sns.histplot(descriptors_per_image, bins=30, kde=True, color="blue")
    plt.title("Distribution of SIFT Descriptors Per Image")
    plt.xlabel("Number of Descriptors")
    plt.ylabel("Frequency")
    plt.show()

def plot_pca(features, labels):
    """Perform PCA and visualize the 2D feature space."""
    pca = PCA(n_components=2)
    features_reduced = pca.fit_transform(features)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=features_reduced[:, 0], y=features_reduced[:, 1],
        hue=labels, palette='coolwarm', alpha=0.7
    )
    plt.title("PCA Visualization of SIFT Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Class", labels=["Benign", "Malignant"])
    plt.show()

# Label Distribution
print("Visualizing Class Label Distribution...")
plot_label_distribution(labels, "Class Distribution in Dataset")

# Descriptor Distribution
print("Visualizing SIFT Descriptor Distribution...")
plot_descriptor_distribution(features, labels)

# PCA Visualization
print("Performing PCA and Visualizing Feature Space...")
plot_pca(features, labels)
