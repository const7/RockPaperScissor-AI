# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np


"""Preparing our Data"""

DATA_PATH = sys.argv[1] # Path to folder containing data

shape_to_label = {"rock": np.array([1., 0., 0.]), "paper": np.array([0., 1., 0.]), "scissor": np.array([0., 0., 1.])}

imgData = list()
labels = list()

for dr in os.listdir(DATA_PATH):
    if dr not in ["rock", "paper", "scissor"]:
        continue
    lb = shape_to_label[dr]
    i = 0
    for pic in os.listdir(os.path.join(DATA_PATH, dr)):
        path = os.path.join(DATA_PATH,dr + "/" + pic)
        img = cv2.imread(path)
        imgData.append([img, lb])
        imgData.append([cv2.flip(img, 1), lb]) #horizontally flipped image
        imgData.append([cv2.resize(img[50:250, 50:250], (300,300)), lb]) # zoom : crop in and resize
        i += 3
    print("{}:{}".format(dr, i))
np.random.shuffle(imgData)

# construct train data
imgData,labels = zip(*imgData)
imgData = np.array(imgData)
labels = np.array(labels)

"""Model"""

from keras.models import Sequential
from keras.layers import Dense,MaxPool2D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet121

"""DenseNet"""

densenet = DenseNet121(include_top=False, weights="imagenet", classes=3, input_shape=(300, 300, 3))
densenet.trainable=True

def genericModel(base):
    model = Sequential()
    model.add(base)
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["acc"])
    return model

dnet = genericModel(densenet)

checkpoint = ModelCheckpoint(
    "model.h5", 
    monitor="val_acc", 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=True,
    mode="auto"
)

es = EarlyStopping(patience=3)

history = dnet.fit(
    x=imgData,
    y=labels,
    batch_size = 16,
    epochs=8,
    callbacks=[checkpoint, es],
    validation_split=0.2
)

dnet.save_weights("model.h5")

with open("model.json", "w") as json_file:
    json_file.write(dnet.to_json())
