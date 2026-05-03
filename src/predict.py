import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("model/fruit_model.h5")

classes = ['apple', 'banana', 'orange']

img = cv2.imread("test_image.jpg")
img = cv2.resize(img, (100, 100))
img = img / 255.0
img = np.reshape(img, (1, 100, 100, 3))

prediction = model.predict(img)
print("Predicted:", classes[np.argmax(prediction)])
