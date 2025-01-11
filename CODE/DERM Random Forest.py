import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

# Load the SIFT features and labels from the .npz file
data = np.load(r"C:\Users\dalla\Desktop\ML PROJECT\DERM Project\Dataset-6k\new_sift_features.npz")
features = data['features']  # Shape: (n_samples, n_features)
labels = data['labels']      # Shape: (n_samples,)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale the features (optional for Random Forest, but ensures consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# Train with progress bar
print("Training Random Forest Model...")
for i in tqdm(range(1, rf_model.n_estimators + 1), desc="Building Trees", unit="tree"):
    rf_model.set_params(n_estimators=i)  # Incrementally increase the number of trees
    rf_model.fit(X_train_scaled, y_train)

# Make predictions
print("Making Predictions...")
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
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
