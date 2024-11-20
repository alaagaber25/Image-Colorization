import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image

# Load the trained model from the pickle file
with open('model/keras_model3.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Image Colorization")

# Upload an image
uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    image = np.array(Image.open(uploaded_file).convert('L'))  # Grayscale image

    # Display the original grayscale image
    st.subheader("Grayscale Image")
    st.image(image, channels="L",width=300)

    # Preprocess the image for your model if required
    image_resized = cv2.resize(image, (224, 224)) / 255.0  # Example: Resize and normalize
    image_resized = np.expand_dims(image_resized, axis=[0, -1])  # Add batch and channel dimensions

    # Perform colorization using the loaded model
    colorized_image = model.predict(image_resized)[0]

    # Post-process the result if necessary (e.g., scaling back pixel values)
    colorized_image = (colorized_image * 255).astype(np.uint8)

    # Display the colorized image
    st.subheader("Colorized Image")
    st.image(colorized_image, width=300)
