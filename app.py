import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁")

st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.markdown("""
This app uses a **Deep Learning Model (ResNet50)** to detect signs of pneumonia in chest X-ray images.
Upload an image to get a prediction.
""")

# 2. LOAD THE MODEL (Cached so it doesn't reload every time you click a button)
@st.cache_resource
def load_pneumonia_model():
    # Make sure this matches the name of the file you saved earlier!
    model = load_model('models/resnet50_pneumonia_model.h5')
    return model

# Show a loading spinner while model loads
with st.spinner('Loading Model...'):
    model = load_pneumonia_model()

# 3. IMAGE PREPROCESSING FUNCTION
def preprocess_image(uploaded_file):
    # Load the image with the target size (224, 224) matches our model input
    img = image.load_img(uploaded_file, target_size=(224, 224))
    
    # Convert image to a numpy array (224, 224, 3)
    img_array = image.img_to_array(img)
    
    # Add a fourth dimension (batch size) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Use the same preprocessing function as training!
    processed_img = preprocess_input(img_array)
    
    return processed_img, img

# 4. FILE UPLOADER UI
uploaded_file = st.file_uploader("Choose a Chest X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Process the image
    processed_img, original_img = preprocess_image(uploaded_file)
    
    # Display the uploaded image
    st.image(original_img, caption="Uploaded X-Ray", use_container_width=True)
    
    # 5. MAKE PREDICTION
    if st.button("Analyze Image"):
        prediction = model.predict(processed_img)
        
        # The output is a probability (0 to 1)
        probability = prediction[0][0]
        
        # Determine class based on probability
        # Remember: 0 = Normal, 1 = Pneumonia
        if probability > 0.5:
            st.error(f"⚠️ **PNEUMONIA DETECTED** (Confidence: {probability:.2%})")
            st.markdown("The model has detected patterns consistent with pneumonia.")
        else:
            st.success(f"✅ **NORMAL** (Confidence: {(1-probability):.2%})")
            st.markdown("The model did not detect signs of pneumonia.")