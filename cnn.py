import tensorflow as tf
import random
import numpy
import cv2
from PIL import Image
from pathlib import Path
import os


training_data_path = Path("/home/willi/validation_data")
training_image_paths = list()

for path in training_data_path.glob("**/*"):
    training_image_paths.append(path.as_posix().__str__())

validation_data_path = Path("/home/willi/validation_data")
test_data_paths = Path("/home/willi/test_data")
validation_image_paths = list(validation_data_path.glob("**/*"))
test_image_paths = list(test_data_paths.glob("**/*"))
label_array = ["DIS","ANG", "FEA", "SAD", "SUR", "JOY", "NEU"]
label_to_index = dict((name, index) for index,name in enumerate(label_array))

def getLabelDict(image_paths):


    all_image_labels = [label_to_index[Path(path).absolute().name[0:3]]
                        for path in image_paths]
    return all_image_labels

def getLabelList(image_paths):
    all_img_labels = list()
    for path in image_paths:
        all_img_labels.append(Path(path).absolute().name[0:3])
    return all_img_labels


def preProcessPath(path):
    return path.absolute().name


def get_ds(data_path):
    image_paths= list()
    for path in list(data_path.glob("**/*")):
        image_paths.append(path.absolute())

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, getLabelList(image_paths)))
    for path in image_paths:
        dataset.map(getPic(path))

    return dataset

def getPic(path):
    image = Image.open(path).convert('RGB')
    image = image.resize((256,256,3))
    array = numpy.array(image.getdata())
    array = array.reshape((256,256,3))
    return array


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



img_generator = tf.keras.preprocessing.image.ImageDataGenerator()
train_iterator = img_generator.flow_from_directory(Path("/home/willi/test_data"), target_size=(256,256), color_mode='rgb', shuffle=True)

img_generator2 = tf.keras.preprocessing.image.ImageDataGenerator()
val_iterator = img_generator.flow_from_directory(Path("/home/willi/validation_data"), target_size=(256,256), color_mode='rgb', shuffle=True)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256,256,3)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_iterator, epochs=1, steps_per_epoch=3,validation_data=val_iterator)
