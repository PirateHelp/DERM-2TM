HOW TO RUN

1. CROP AND RESIZE.py - (Raw images are preprocesed to remove noise and enlarge the skin lesion for better results) Raw images are now placed in new cropped folders
2. SIFT Feature Extraction DERM.py - Loads cropped image folders, converts to grayscale, resizes images, extracts keypoints with SIFT features, saves new file with keypoints
3. SIFT Keypoint Visualization DERM.py - Code for visualizing and saving images with the keypoints on the images, great for visual processing.
4. Choose a model - CNN, KNN, LSTM, etc. Loads the cropped images folder and model uses own train test split within code. Prints F1 Score, Accuracy, Precision, Etc.

   NOTE: CNN model uses cropped images while other models use SIFT features.

The models are evaluated using accuracy, precision, recall, and confusion matrices.
