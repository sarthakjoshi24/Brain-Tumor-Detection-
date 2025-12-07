import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('BrainTumor10Epochs.h5')
    return model

model = load_model()

def preprocess_image(image):
    img = image.resize((64, 64))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main():
    st.title("Brain Tumor Detection App")

    # User uploads an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Diagnose"):
            processed_img = preprocess_image(image)
            result = model.predict(processed_img)

            if result[0][0] >= 0.5:
                st.error("Brain Tumor Detected")
            else:
                st.success("No Brain Tumor Detected")

if __name__ == "__main__":
    main()
