
# 🫁 Pneumonia Detection from Chest X-Rays

## 📌 Project Overview

This project is a Deep Learning application designed to classify Chest X-Ray images as either **Normal** or  **Pneumonia** . It utilizes **Transfer Learning** with a fine-tuned **ResNet50** architecture to achieve high sensitivity (Recall), ensuring minimal false negatives in medical screening. The model is deployed as an interactive web application using  **Streamlit** .

## 🚀 Key Features

* **Deep Learning Model:** Fine-tuned ResNet50 (pre-trained on ImageNet) for binary classification.
* **Data Augmentation:** Implemented random rotation, zooming, and shearing to improve model generalization and prevent overfitting.
* **High Sensitivity:** Optimized for medical context, achieving a **Recall of 97%** for Pneumonia cases.
* **Interactive UI:** User-friendly web interface built with Streamlit for real-time image analysis.

## 🛠️ Tech Stack

* **Language:** Python
* **Frameworks:** TensorFlow, Keras
* **Web Deployment:** Streamlit
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

## 📂 Dataset

The model was trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia "null") dataset.

* **Classes:** Normal vs. Pneumonia
* **Training Images:** 5,216
* **Test Images:** 624

## ⚙️ Installation & Usage

### 1. Clone the Repository

```
git clone https://github.com/keshavk215/Pneumonia-Detection-from-Chest-X-Ray-Images.git
cd Pneumonia-Detection-from-Chest-X-Ray-Images

```

### 2. Install Dependencies

```
pip install -r requirements.txt

```

### 3. Run the Web App

```
streamlit run app.py

```

The application will open in your browser at `http://localhost:8501`.

## 🧠 Methodology

### 1. Data Preprocessing

* Images resized to `224x224` pixels to match ResNet50 input requirements.
* **Augmentation:** Applied only to training data to simulate variations in X-ray positioning.
* **Preprocessing:** Used ResNet's specific preprocessing function (mean subtraction/scaling) rather than simple 0-1 normalization.

### 2. Model Architecture (Transfer Learning)

Instead of training from scratch, I leveraged  **ResNet50** :

* **Base Model:** Frozen ResNet50 weights (ImageNet).
* **Fine-Tuning:** Unfroze the last 10 layers of the base model to adapt high-level feature extraction specifically for X-ray textures.
* **Custom Head:** Added GlobalAveragePooling, Dense layers, and Dropout (0.5) to prevent overfitting.
* **Optimizer:** Adam with a low learning rate (`1e-5`) during fine-tuning to preserve pre-trained knowledge.

## 📊 Results & Evaluation

The model was evaluated on the unseen Test set (624 images).

| **Metric**             | **Score** | **Note**                                          |
| ---------------------------- | --------------- | ------------------------------------------------------- |
| **Accuracy**           | **89%**   | Overall correctness                                     |
| **Recall (Pneumonia)** | **97%**   | Critical for medical screening (minimizes missed cases) |
| **Precision**          | **87%**   | Acceptable trade-off for higher recall                  |

### Confusion Matrix

* **True Positives:** 378 (Correctly identified Pneumonia)
* **False Negatives:** 12 (Missed Pneumonia cases)
* **False Positives:** 55 (Normal flagged as Pneumonia)
* **True Negatives:** 179 (Correctly identified Normal)
