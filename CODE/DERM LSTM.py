import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Paths to the folders
benign_path = r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\Dataset-6k\benign cropped"
malignant_path = r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\Dataset-6k\malignant cropped"

# Image parameters
image_size = (128, 128)  # Resize all images to 128x128
sequence_length = 128    # Treat each row of the image as a sequence step

# Load and preprocess images
def load_images(folder, label):
    images = []
    labels = []
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is not None:
            img = cv2.resize(img, image_size)  # Resize
            images.append(img)
            labels.append(label)  # Assign the label (0: benign, 1: malignant)
    return np.array(images), np.array(labels)

# Load benign and malignant images
benign_images, benign_labels = load_images(benign_path, label=0)
malignant_images, malignant_labels = load_images(malignant_path, label=1)

# Combine and shuffle data
X = np.vstack((benign_images, malignant_images))
y = np.hstack((benign_labels, malignant_labels))

# Normalize pixel values to [0, 1]
X = X / 255.0

# Reshape images into sequences for LSTM
X = X.reshape(X.shape[0], sequence_length, -1)  # Shape: (samples, sequence_length, features_per_step)

# One-hot encode labels for categorical classification
y = to_categorical(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))  # 2 classes: benign and malignant

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the LSTM model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification Report
print("Classification Report:\n", classification_report(y_true_classes, y_pred_classes))

# Plot the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
