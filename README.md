# AI-vs-REAl-Image-Classifier

## Introduction:

This project aims to develop a machine learning model that can classify images as either AI-generated or real. The classifier uses a convolutional neural network (CNN) to differentiate between synthetic images generated by AI algorithms and real photographs taken by cameras.

## Data Preprocessing:

To ensure uniformity among the images, we resized all of them to a consistent dimension of 32x32 pixels. Each image is represented as an array with the shape (32, 32, 3), corresponding to the RGB color channels.
Each 2D matrix within the array separately manages the red, green, and blue pixel components.

## Data Collection:

I took help of the CIFAKE dataset(https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) available on kaggle along with images from web scraping using selenium, beautifulsoup and pygoogle_image modules of python(In the file web_scraping.py).
Hence the model was trained on 100831 Images. It was tested on 20000 images consisting of both real and fake images.

## Why CNN is Used in This Project:

Convolutional Neural Networks (CNNs) are employed in this project due to their superior ability to handle image data. CNNs excel at recognizing patterns and features in images through their convolutional layers, which apply various filters to the input images to detect edges, textures, and other important details. This makes CNNs particularly well-suited for image classification and recognition tasks. Their architecture, which includes pooling layers to reduce dimensionality and fully connected layers for classification, enables efficient processing and accurate interpretation of visual data. The inherent ability of CNNs to learn spatial hierarchies of features further enhances their effectiveness in tasks involving image data.

## Result:

The model achieved a F1-score of 0.93 on the test dataset, demonstrating its ability to accurately classify images as either AI-generated or real.

## Note:

The file deploy.py contains the code for web application generated using streamlit.Clone this repo and just run deploy.py (using python -m streamlit run js.py). This will open the website.
Check out the live version here - "https://chandini-ai-vs-real-image-classifier-28.streamlit.app/"

## Done By:

Chandini Adapa
