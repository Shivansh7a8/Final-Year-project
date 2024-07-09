import os
import json
from PIL import Image #to show the images within the classes 

import numpy as np  #lib used for mathematical operation
import tensorflow as tf  #open-source ml lib used for training the model
import streamlit as st  #open-source web framework for developing ML based web-app


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/tf_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('Botanic Image Analyzer')
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
# img=st.camera_input("Take a pic")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((200, 200))
        st.image(resized_img)

    with col2:
        if st.button('DETECT'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
    
st.divider()

# Display multiple images in a row
images = [
    "images/apples1.jpg" ,  # Replace these with the paths to your images
    "images/grape.jpg" ,
    "images/corn1.jpg" ,
    "images/orange1.jpg" ,
    "images/potato.jpg" ,
    "images/tomato.jpg" ,
    "images/rb.jpg",
    "images/cherry.jpg",
    "images/peach1.jpg"
    ]
st.image(images, caption=['Apple','Grapes','Corn','Orange','Potato','Tomato','Raspberry','cherry','peach'], width=200)
st.write('Many more')
# st.balloons()
# st.snow()
