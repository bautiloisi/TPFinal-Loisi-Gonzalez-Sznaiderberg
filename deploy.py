import streamlit as st
import tensorflow as tf
import requests
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
@st.cache_resource
def load_mask_detection_model():
    model_path = 'model.h5'
    model = load_model(model_path)
    return model

# Function to preprocess the image
@st.cache_data
def load_image(image_path):
    response = requests.get(image_path, stream=True)
    if response.status_code == 200:
        img = Image.open(response.raw)
        img = img.resize((128, 128))  # Resize the image to match the model's expected sizing
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img, img_array
    else:
        return None, None

# Streamlit app
def main():
    st.title("¿Tiene Barbijo?")

    # Write Text
    st.write("Ingrese el link de una imagen sobre la cara de una persona para comprobar si tiene o no barbijo")

    # Load the model
    mask_detection_model = load_mask_detection_model()

    # Get image URL from user
    image_path = st.text_input("Enter Image URL to classify...")

    # Get image from URL and predict
    if image_path:
        try:
            st.write("Predicting Class...")
            with st.spinner("Classifying..."):
                img, img_array = load_image(image_path)
                if img is not None and img_array is not None:
                    st.image(img, use_column_width=True, caption="Input Image")  # Display the input image
                    pred = mask_detection_model.predict(img_array)
                    st.write(pred[0][0])
                    pred_class = "tiene Barbijo" if pred[0][0] > 0.01 else "no tiene Barbijo"
                    st.write("La persona de la foto", pred_class)
                else:
                    st.write("Invalid Image URL")
        except:
            st.write("Error processing the image")

if __name__ == '__main__':
    main()
