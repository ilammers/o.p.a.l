import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Load the image using PIL
image = cv2.imread('./adv3.jpg')

desired_height = 30
desired_width = 30
channels = 3

interpreter = tf.lite.Interpreter(model_path="convertedModel")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess the image
# Resize the image to match the input shape of your model
input_shape = (desired_height, desired_width)  # Specify the desired input shape
image_fromarray = Image.fromarray(image, 'RGB')
resize_image = image_fromarray.resize(input_shape)

# Convert the image to a numpy array
image_array = np.array(resize_image)

# Normalize the image data if required (e.g., scale pixel values to [0, 1])
# image_array = image_array.astype(np.float32) / 255.0  # Assuming pixel values are in [0, 255] range
image_array = (np.float32(image_array)) / 255.0

# Expand dimensions if necessary to match the expected input shape of your model
input_data = np.expand_dims(image_array, axis=0)  # Add batch dimension

print(input_data.shape)

# Set the input tensor of your TensorFlow interpreter
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor(s)
output_data = interpreter.get_tensor(output_details[0]['index'])

# For classification tasks, assuming the output tensor contains class probabilities
# Get the index of the predicted class (assuming single class prediction)
predicted_class_index = np.argmax(output_data[0])

print(predicted_class_index)
