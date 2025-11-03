# 🧠 Brain Tumor MRI Classification (Kaggle Notebook)

This repository contains a Jupyter Notebook (`.ipynb`) for the multiclass classification of brain tumors from MRI scans. The model uses a transfer learning approach with **VGG16** to classify images into four categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor.

The entire pipeline—from data loading and augmentation to model training and evaluation—is contained within the notebook, making it easy to follow the process step-by-step.

## 📊 Evaluation & Visualizations

The notebook automatically generates a comprehensive evaluation of the model's performance, including:

* **Training History:** Plots for training vs. validation accuracy and loss.
* **Classification Report:** Detailed precision, recall, and F1-scores for each tumor type.
* **Confusion Matrix:**  A heatmap to visualize correct and incorrect predictions.
* **ROC/AUC Curves:** Multi-class ROC curves to show the model's diagnostic ability.
* **Prediction Visualization:** A grid of test images with their true and predicted labels.

## 📁 Dataset

This model is trained on the **Brain Tumor MRI Dataset** available on Kaggle.

* **Link:** [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* **Note:** The notebook is pre-configured to read data from the standard Kaggle input directory (`/kaggle/input/brain-tumor-mri-dataset`).

## ✨ Features

* **Model:** VGG16 (pre-trained on ImageNet) with fine-tuning.
* **Data Handling:** A custom `data_generator` loads images in batches to prevent memory-related crashes.
* **Data Augmentation:** Real-time augmentation (Brightness, Contrast, Sharpness, and Horizontal Flips) to create a more robust model.
* **Robust Training:** Implements `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` to save the best model and prevent overfitting.
* **Clear Structure:** The code is organized into a `BrainTumorClassifier` class, making the methodology clear and reusable.

## 🛠️ Technologies Used

* Jupyter Notebook
* TensorFlow / Keras
* Scikit-learn
* NumPy
* Matplotlib & Seaborn
* Pillow (PIL)

## 🚀 How to Run


1.  **Upload:** Sign in to Kaggle, go to "Code," and click "New Notebook." Upload this `.ipynb` file.
2.  **Add Data:** In the notebook editor, click "+ Add data" in the right-hand panel. Search for the "Brain Tumor MRI Dataset" and add it.
3.  **Run:** The notebook should run as-is, as the file paths (`/kaggle/input/...`) will match.



## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
