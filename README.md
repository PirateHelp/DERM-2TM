# DERM-2TM
This project is focused on classifying skin lesions as malignant or benign using machine learning. It involves preprocessing images, extracting features, training models, and evaluating their performance.

Objective: Classify benign and malignant skin lesion images using machine learning models.

Dataset:
6,000 cropped dermatoscopic images.
Noise (e.g., rulers, hair) reduced during preprocessing.

Key Steps:

Feature Extraction:
Used SIFT for improved feature representation.

Modeling:
Models: SVM, KNN, Random Forest, Decision Tree, XGBoost, CNN.


How It Works
1.	Preprocessing:
o	Images are cropped to remove noise like rulers or hair.
o	Sizes are normalized for consistency.

3.	Feature Extraction:
o	SIFT (Scale-Invariant Feature Transform) is used to extract important features for traditional machine learning models.

5.	Model Training:
o	Models like SVM, KNN, Random Forest, Decision Tree, XGBoost, and CNN are trained on the data.

7.	Evaluation:
o	The models are evaluated using accuracy, precision, recall, and confusion matrices.
