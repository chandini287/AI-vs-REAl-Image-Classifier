import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('Model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((32,32))  
    img_array = np.array(image)
    img_array = img_array / 255
    return img_array

# Function to make a prediction
def predict(image):
    processed_image = preprocess_image(image)
    test = np.array([processed_image])
    pred = model.predict(test)
    ind = np.argmax(pred[0])
    if ind == 0:
        return "AI generated Image"
    elif ind == 1:
        return "REAL Image"

# Streamlit app
def main():
    # CSS styling to set background image and improve text readability
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://wallpaperaccess.com/full/1846972.jpg");
            background-size: cover;
            background-position: center;
        }
        .css-1aumxhk {  /* Streamlit sidebar background */
            background: white;
        }
        .css-1d391kg { /* Applies to the main area */
            background: rgba(255, 255, 255, 0.8); /* Semi-transparent white overlay for readability */
            border-radius: 10px;
            padding: 2rem;
            margin: auto; /* Center the main content */
            max-width: 700px; /* Control the width of the main content */
            color: black; /* Set main text color to black */
            text-align: center; /* Center-align text */
        }
        h1, h2, h3 {
            color: black; /* Keep headers in black */
        }
        p {
            color: rgba(255, 255, 255, 0.9); /* Set paragraph text to light gray/white */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # App title
    st.title('AI vs REAL Image Classifier')

    # Sidebar page selection
    pages = ['Home', 'Predictor']
    selected_page = st.sidebar.radio('Select a page', pages)

    # Home page content
    if selected_page == 'Home':
        st.header('Welcome to the AI Image Classifier!')
        st.write('This app allows you to classify images as real or AI generated.')

    # Predictor page content
    elif selected_page == 'Predictor':
        st.header('Image Predictor')
        uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write('')

            if st.button('Predict'):
                label = predict(image)
                st.write(f'Prediction: {label}')

if __name__ == '__main__':
    main()
