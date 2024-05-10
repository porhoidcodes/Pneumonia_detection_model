import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('pneumonia_detection_model.h5')

# Define a function to make predictions
def predict_pneumonia(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Make a prediction
    prediction = model.predict(image)

    # Postprocess the prediction
    if prediction[0][0] > prediction[0][1]:
        return "The patient does not have pneumonia."
    else:
        return "The patient has pneumonia."

# Create a Gradio interface
iface = gr.Interface(
  fn=predict_pneumonia,
  inputs="image",
  outputs="text",
  title="Pneumonia Detection",
  description="Upload an X-ray image to detect pneumonia",
  theme="ysharma/steampunk",  # Replace with desired theme URL
)


# Launch the interface
iface.launch()