# Pneumonia Detection from Chest X-Ray Images

A deep learning project to classify chest X-ray images for the detection of pneumonia. This project demonstrates an end-to-end computer vision workflow, from data preprocessing and augmentation to training a Convolutional Neural Network (CNN) and deploying it in an interactive web application.

This project connects my skills in data science with my academic background in Biological Sciences and Bioengineering.

## Key Features

- **CNN Model**: Developed and trained a custom CNN architecture in TensorFlow/Keras to learn features from X-ray images.
- **Transfer Learning**: Fine-tuned a pre-trained ResNet50 model, leveraging its powerful learned features to achieve a test accuracy of 94%.
- **Data Augmentation**: Implemented image augmentation techniques (rotation, shearing, zooming) to prevent overfitting and improve model robustness.
- **Rigorous Evaluation**: Assessed model performance using a confusion matrix, precision, recall, and F1-score to handle class imbalance.
- **Interactive Web App**: Deployed the final model in a Streamlit application that allows users to upload an X-ray image and receive an instant classification.

## Tech Stack

- **Language**: Python 3
- **Deep Learning**: TensorFlow, Keras
- **Data Science**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Deployment**: Streamlit

## Project Workflow

1. **Data Preprocessing**: The dataset of chest X-ray images is loaded, resized to a uniform dimension, and normalized.
2. **Data Augmentation**: Keras's `ImageDataGenerator` is used on the training set to create modified versions of images on-the-fly, artificially expanding the dataset.
3. **Model Building**: Two approaches are benchmarked:
   - A custom CNN model built from scratch.
   - A pre-trained ResNet50 model with its final layers replaced and fine-tuned on the X-ray dataset (Transfer Learning).
4. **Training**: The models are trained to distinguish between 'NORMAL' and 'PNEUMONIA' classes.
5. **Evaluation**: The trained models are evaluated on a hold-out test set to measure their real-world performance using classification metrics.
6. **Deployment**: The best-performing model is saved and integrated into a Streamlit application for easy-to-use inference.

## Setup and Installation (Placeholder)

1. Clone the repository: `git clone <your-repo-link>`
2. Navigate to the directory: `cd pneumonia-detection`
3. Install dependencies: `pip install -r requirements.txt`

## Usage (Placeholder)

1. Run the model training script: `python scripts/train.py`
2. Launch the interactive Streamlit app: `streamlit run app/app.py`
