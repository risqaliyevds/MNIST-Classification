import fastai
import matplotlib.pyplot as plt
import streamlit as st
from fastai.vision.all import *
import tensorflow as tf
#import pathlib
import plotly.express as px
#from PIL import Image

# Title of project
st.title('Single digit classificator model!')
st.warning('Please join only single digit photo', icon="⚠️")

# Function for resize image to dashboard
def image_square(file, size):
    # Open the image
    image = Image.open(file)
    # Get the width and height of the image
    width, height = image.size
    # Define the crop size
    crop_size = min(width, height)
    # Calculate the left, upper, right, and lower coordinates
    left = (width - crop_size) / 2
    upper = (height - crop_size) / 2
    right = (width + crop_size) / 2
    lower = (height + crop_size) / 2
    # Define the crop box
    crop_box = (left, upper, right, lower)
    # Crop the image
    cropped_image = image.crop(crop_box)
    # Resize image to show on dashboard
    img = cropped_image.resize((size, size)).convert('L')
    return img

# Function resize image into 28*28 px
def resize(image):
    img = image.resize((28, 28))
    return np.asarray(img)

dict_of_numbers = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: "Four",
                   5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: 'Nine'}
# Streamlit title
file = st.file_uploader('Load Image here', type = ['png', 'jpeg', 'gif', 'svg', 'jpg'])

if file:
    image = image_square(file, 400)
    # Show image on dashboard
    st.image(image)
    # Convert into arrays
    array = resize(image)
    # Load model
    model = tf.keras.models.load_model('model.h5')
    # Predicting
    prediction = model.predict(array.reshape(-1, 28, 28, 1))
    prediction_arg = np.argmax(prediction, axis=1)
    # Info about prediction
    st.success(f'Prediction is: {dict_of_numbers[prediction_arg[0]]}')
    st.info(f'Probability: {float(prediction[0][prediction_arg]) * 100:.2f} %')

    # Ploting
    st.subheader('Probability for each number')
    fig = px.bar(y=prediction[0] * 100, x=np.arange(10),
                 labels={'y':'Probabilities', 'x': 'Labels'})
    st.plotly_chart(fig)


# Display the correct images
st.warning('To get higer accuracy please use paint and black background!', icon="⚠️")
st.success('Right formats', icon="✅")

img1 = image_square("examples/one.jpg", 230)
img2 = image_square("examples/seven.jpg", 230)
img3 = image_square("examples/three.jpg", 230)

images = [img1, img2, img3]
st.image(images, use_column_width=False, caption=["Right format"] * len(images))

# Display the correct images
st.error('Uncorrect photo samples', icon="🚨")

img1_w = image_square("examples/one_white.jpg", 230)
img2_w = image_square("examples/seven_white.jpg", 230)
img3_w = image_square("examples/three_white.jpg", 230)

images_w = [img1_w, img2_w, img3_w]
st.image(images_w, use_column_width=False, caption=["Wrong format"] * len(images))
