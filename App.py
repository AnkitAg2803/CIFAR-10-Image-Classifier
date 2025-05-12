import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
model = tf.keras.models.load_model('best_cifar10_model.h5')

st.title("CIFAR-10 Image Classifier")
st.write("Upload a 32x32 image to classify it into one of the 10 CIFAR-10 categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and show the image
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(image) / 255.0  # Normalize
    if img_array.shape == (32, 32):  # Grayscale image
        img_array = np.stack((img_array,) * 3, axis=-1)  # Convert to 3 channels

    if img_array.shape == (32, 32, 3):
        img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)

        # Predict
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        class_name = class_names[class_index]

        st.success(f"Predicted Class: **{class_name}**")
    else:
        st.error("Invalid image shape. Please upload a valid 32x32 image.")
