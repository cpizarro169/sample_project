import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mymodel_new.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]


sunflower_path = 'test_2.jpg'
img_height = 180
img_width = 180

# load the image 
img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)

# add extra dimension
# as the original shape was None, 180,180, 3
# as tflite model accepts input of shape 1, 180,180,3 add extra dimension in the start
img_array = tf.expand_dims(img_array, 0) 
#print('Shape of input: ',img_array.shape)
# display the image
#import matplotlib.pyplot as plt
#plt.imshow(np.squeeze(img_array,0).astype(int))

# get prediction from tf model
#tf_output = model.predict(img_array)
#tf_output

# set the input values for the model
interpreter.set_tensor(input_details[0]['index'], img_array)
# run the model (forward pass)
interpreter.invoke() 
# get the outputs of the model
output_data = interpreter.get_tensor(output_details[0]['index'])

label_path = "rooms_labels.txt"
labels = load_labels(label_path)
#print(labels)

#print("Prediction of tf model: ", tf_output.argmax())
print("Prediction of tf lite model: ", labels[output_data.argmax()])

print("Prediction of tf lite model: ", labels[output_data.argmax()],
" with confidence : ",output_data[0][output_data.argmax()] )

my_next = np.argsort(-output_data, axis=1)[0,1]
print("Runner up prediction of tf lite model: ", labels[my_next],
" with confidence : ",output_data[0][my_next] )


