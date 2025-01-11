import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

# Load the SIFT features and labels from the .npz file
data = np.load(r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\Dataset-6k\new_sift_features.npz")
features = data['features']  # Shape: (n_samples, n_features)
labels = data['labels']      # Shape: (n_samples,)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)  # Linear kernel for simplicity
svm_model.fit(X_train_scaled, y_train)

# Predict with progress bar
print("Making Predictions...")
y_pred = []

# Wrap `enumerate` around `tqdm` for proper progress bar display
for i, sample in enumerate(tqdm(X_test_scaled, desc="Predicting", unit="sample")):
    pred = svm_model.predict([sample])  # Predict one sample at a time
    y_pred.append(pred[0])

# Evaluate the model
y_pred = np.array(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
