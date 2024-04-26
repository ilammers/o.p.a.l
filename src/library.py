import tensorflow as tf
from PIL import Image
import numpy as np
import subprocess 
import vlc
import time
import cv2
import pyaudio

def capture(filename):
    subprocess.run(["libcamera-still", "-o", filename + ".jpg"])

def playMedia(audiopath):
    media_player = vlc.MediaPlayer() 
    
    media = vlc.Media(audiopath) 
  
    media_player.set_media(media) 
  
    media_player.video_set_scale(0.6) 
  
    media_player.play() 
  
    time.sleep(100) 
  
    value = media_player.audio_output_device_enum() 
  
    print("Audio Output Devices: ") 
    print(value) 

def generateSinewave(paramf, paramf1):
    p = pyaudio.PyAudio()

    volume = 1.0  # range [0.0, 1.0]
    fs = 44100  # sampling rate, Hz, must be integer
    duration = 8.0  # in seconds, may be float
    f = paramf  # sine frequency, Hz, may be float
    f1 = paramf1 #Sine frequency 2, Hz, may be float

    # generate samples, note conversion to float32 array
    # add two sine functions together to get a chord
    samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs) + np.sin(2 * np.pi * np.arange(fs * duration) * f1 / fs)).astype(np.float32)

    # per @yahweh comment explicitly convert to bytes sequence
    output_bytes = (volume * samples).tobytes()

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    # play. May repeat with different volume values (if done interactively)
    start_time = time.time()
    stream.write(output_bytes)
    print("Played sound for {:.2f} seconds".format(time.time() - start_time))

    stream.stop_stream()
    stream.close()

    p.terminate()

def predict(imagepath):
    # Load the image using cv2 - required for interpreting RGB images
    image = cv2.imread(imagepath)

    desired_height = 30
    desired_width = 30

    interpreter = tf.lite.Interpreter(model_path="convertedModel")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize the image to match the input shape of your model
    input_shape = (desired_height, desired_width)  # Specify the desired input shape
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize(input_shape)

    # Convert the image to a numpy array
    image_array = np.array(resize_image)

    # Normalize the image data if required (e.g., scale pixel values to [0, 1])
    # image_array = image_array.astype(np.float32) / 255.0 
    image_array = (np.float32(image_array)) / 255.0

    # Expand dimensions if necessary to match the expected input shape of your model
    input_data = np.expand_dims(image_array, axis=0)  # Add batch dimension

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
