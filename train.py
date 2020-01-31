# we are using the Squeezenet model. so if you are using keras version greater than 2.2 then:-
# replace 'keras.applications.imagenet_utils' to 'keras_applications.imagenet_utils' from Squeezenet model.
# you can find the code of Squeezenet model in the keras directory
# Idk why but the 'warning' in keras Squeezenet model doesnt work as well so remove that.

import cv2
import numpy as np
import tensorflow as tf
from keras_squeezenet import SqueezeNet

from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential

import os

# this is the folder where all the photos will be stored
IMG_SAVE_PATH = 'train-data'

CLASS_MAP = {
    #"up": 0,
    "Down": 0,
    "Left": 1,
    "none": 2,
    "Right" :3
}

NUM_CLASSES = len(CLASS_MAP)


def mapper(val):
    return CLASS_MAP[val]


def get_model():
    model = Sequential([
        SqueezeNet(input_shape=(227, 227, 3), include_top=False), #we are using the  pretrained squeezenet model
        Dropout(0.5),
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model


# load images from the directory
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

'''
dataset = [
    [[...], 'rock'],
    [[...], 'paper'],
    ...
]
'''
data, labels = zip(*dataset)
labels = list(map(mapper, labels))


'''
labels: rock,paper,paper,scissors,rock...
one hot encoded: [1,0,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]...
'''

# one hot encode the labels
labels = np_utils.to_categorical(labels)

# define the model
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# start training
model.fit(np.array(data), np.array(labels), epochs=7)

# save the model for later use
model.save("car-racing-model.h5")
