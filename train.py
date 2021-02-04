# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf


DATA_PATH = "data"
MODEL_PATH = "model"
MODEL_STRUCT = "model.json"
MODEL_WEIGHT = "modelweight.h5"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# avoid "CUBLAS_STATUS_ALLOC_FAILED"
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

"""Preparing our Data"""

def getData(filepath):
    shape_to_label = {"rock": np.array([1., 0., 0.]), "paper": np.array([0., 1., 0.]), "scissor": np.array([0., 0., 1.])}

    imgData = list()
    labels = list()

    for dr in os.listdir(os.path.join(filepath)):
        if dr not in ["rock", "paper", "scissor"]:
            continue
        lb = shape_to_label[dr]
        i = 0
        for pic in os.listdir(os.path.join(filepath, dr)):
            path = os.path.join(filepath, dr, pic)
            img = cv2.imread(path)
            imgData.append([img, lb])
            # imgData.append([cv2.flip(img, 1), lb]) #horizontally flipped image
            # imgData.append([cv2.resize(img[50:250, 50:250], (300,300)), lb]) # zoom : crop in and resize
            i += 1
        print("{}: {}".format(dr, i))
    np.random.shuffle(imgData)

    # reshape data to model input
    imgData,labels = zip(*imgData)
    imgData = np.array(imgData)
    labels = np.array(labels)

    return imgData, labels

trainX, trainY = getData(DATA_PATH)

"""Model"""

from keras.models import Sequential
from keras.layers import Dense,MaxPool2D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet121

"""DenseNet"""
# load a pre-trained weights
densenet = DenseNet121(include_top=False, weights="imagenet", classes=3, input_shape=(300, 300, 3))
densenet.trainable = True

# define model architecture
def genericModel(base):
    model = Sequential()
    model.add(base)
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["acc"])
    return model

dnet = genericModel(densenet)

# define the configuration required for training
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_PATH, MODEL_WEIGHT), 
    monitor="val_acc", 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=True,
    mode="auto"
)
es = EarlyStopping(patience=3)
tb = tf.keras.callbacks.TensorBoard(log_dir="./logs")

# train our model
history = dnet.fit(
    x=trainX,
    y=trainY,
    batch_size=8,
    epochs=8,
    callbacks=[checkpoint, es, tb],
    validation_split=0.2
)

# save model
# dnet.save_weights(os.path.join(MODEL_PATH, MODEL_WEIGHT))
with open(os.path.join(MODEL_PATH, MODEL_STRUCT), "w") as json_file:
    json_file.write(dnet.to_json())
