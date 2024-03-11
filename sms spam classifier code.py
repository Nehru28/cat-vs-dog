import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Function to predict if the image contains a cat or a dog
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # Decode the prediction into readable labels
    decoded_preds = tf.keras.applications.resnet50.decode_predictions(preds, top=1)[0]

    # Check if the prediction is either a cat or a dog
    for i, (_, label, prob) in enumerate(decoded_preds):
        if 'cat' in label or 'dog' in label:
            return f'This image contains a {label} with a probability of {prob:.2f}'

    return 'This image does not contain a cat or a dog'

# Example usage
image_path = 'path_to_your_image.jpg'
print(predict_image(image_path))
