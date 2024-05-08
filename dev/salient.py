import tensorflow as tf
import cv2
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

model = tf.keras.models.load_model('yoloTest.h5')

img_path = './green/avs13.jpeg'
IMG_HEIGHT = 30
IMG_WIDTH = 30

data =[]

try:
    image = cv2.imread(img_path)
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
    data.append(np.array(resize_image))
except:
    print("Error in ")

X_test = np.array(data)
X_test = X_test/255

X_tensor = tf.ragged.constant(X_test).to_tensor()

class_idx = 14

def get_saliency_map(model, image, class_idx):
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)

        classes_x = np.argmax(predictions,axis=1)
        print('Predicted Class Value' + str(classes_x))
        
        loss = predictions[:, class_idx]
    
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)
    
    # take maximum across channels
    gradient = tf.reduce_max(gradient, axis=-1)
    
    # convert to numpy
    gradient = gradient.numpy()
    
    # normaliz between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + keras.backend.epsilon())
    
    return smap

smap = get_saliency_map(model, X_tensor, class_idx)

# plt.imshow(smap[0], alpha=0.6)

blurred = gaussian_filter(smap[0], sigma=5)

def plotSMAP():
    plt.imshow(data[0], cmap='gray', interpolation='none') 
    plt.imshow(blurred, cmap='jet', alpha=0.5*(blurred>0), interpolation='none')

def plotImg(img):
    image = cv2.imread(img)[:,:,::-1]
    plt.imshow(image)
    plt.show()


plotImg(img_path)
plotSMAP()