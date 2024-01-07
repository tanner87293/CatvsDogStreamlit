import streamlit as st
import urllib.request
from fastai.vision.all import *


def label_func(f): return f[0].isupper()


# Load the pre-trained model
model = load_learner('cats_vs_dog.pkl')


# Define a function to make predictions
def predict(image):
    img = PILImage.create(image)  # Use PILImage.create to open the image
    pred_class, pred_idx, outputs = model.predict(img)
    likelihood_is_cat = outputs[1].item()
    if likelihood_is_cat > 0.9:
        return "Cat"
    elif likelihood_is_cat < 0.1:
        return "Dog"
    else:
        return "Not sure... try another picture!"


# Streamlit app title and description
st.title("Cat vs. Dog Classifier by Gamas Chang")
st.write("Upload an image, and I'll tell you whether it's a cat or a dog!")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Make predictions on the uploaded image
    if st.button("Predict"):
        prediction = predict(uploaded_file)
        st.write(prediction)

# Add a footer
st.text("Built with Streamlit and Fastai")