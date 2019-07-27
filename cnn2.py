import tensorflow as tf
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import os
import glob

label_array = ["DIS","ANG", "FEA", "SAD", "SUR", "JOY", "NEU"]
label_to_index = dict((name, index) for index,name in enumerate(label_array))

# Takes as input path to image file and returns
# resized 3 channel RGB image of as numpy array of size (256, 256, 3)
def getPic(img_path):
    return np.array(Image.open(img_path).convert('RGB').resize((256,256),Image.ANTIALIAS))

# returns the Label of the image based on its first 3 characters
def get_label(img_path):
    return Path(img_path).absolute().name[0:3]

# Return the images and corresponding labels as numpy arrays
def get_ds(data_path):
    img_paths = list()
    # Recursively find all the image files from the path data_path
    for img_path in glob.glob(data_path+"/**/*"):
        img_paths.append(img_path)
    images = np.zeros((len(img_paths),256,256,3))
    labels = np.zeros(len(img_paths))

    # Read and resize the images
    # Get the encoded labels
    for i, img_path in enumerate(img_paths):
        images[i] = getPic(img_path)
        labels[i] = label_to_index[get_label(img_path)]

    return images,labels

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(256,256,3),filters=5, kernel_size=10, padding='valid', strides=1),
    tf.keras.layers.Conv2D(filters=5, kernel_size=5, padding='valid', strides=2),
    tf.keras.layers.Conv2D(filters=5, kernel_size=3, padding='valid', strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(13, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(7, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load the train and validation data
train_X, train_y = get_ds("/home/willi/labeled_data")
val_X, val_y = get_ds("/home/willi/validation_data")

# Finally train it
model.fit(train_X,train_y, batch_size=2000, validation_data=(val_X,val_y))

# Predictions
model.predict(val_X)