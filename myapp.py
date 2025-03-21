import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model
model = tf.keras.models.load_model('64x3-CNN.keras')

# Define function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_names = ['No Diabetic Retinopathy', 'Diabetic Retinopathy']
    return class_names[np.argmax(prediction)], prediction[0]

# Streamlit UI
st.title("Diabetic Retinopathy Detection")
st.write("Upload a retinal image to check for diabetic retinopathy.")
st.markdown("### What is Diabetic Retinopathy?")
st.write("Diabetic retinopathy is a diabetes complication that affects the eyes. It is caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). Early detection is crucial to prevent vision loss.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Predict"):
        label, confidence = predict(image)
        st.write(f"**Prediction:** {label}")
        st.progress(float(confidence[np.argmax(confidence)]))
        st.write(f"**Confidence Scores:** No Diabetic Retinopathy: {confidence[0]:.4f}, Diabetic Retinopathy: {confidence[1]:.4f}")
        
        if label == "Diabetic Retinopathy":
            st.warning("Warning: The model predicts signs of diabetic retinopathy. Please consult an eye specialist for further evaluation.")