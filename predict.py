import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# load trained model
model = tf.keras.models.load_model("medical_model.h5")

# image path
img_path = "test_image.png"

# load image
img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# prediction
prediction = model.predict(img_array)

# class names (CHANGE THIS based on your dataset folders)
class_names = ["normal", "pneumonia"]

print("Prediction:", class_names[np.argmax(prediction)])
print("Confidence:", float(np.max(prediction)))