import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Reshape, BatchNormalization, Activation, Concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, SGD
from keras.callbacks import Callback
from keras import backend as K

def generate_model(input_shape=(160, 60, 3)):

    def buildNetwork(input_shape):

        base_network = Sequential([
            Convolution2D(4, kernel_size=6, padding="same", input_shape=input_shape),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(strides=2),

            Dropout(0.3),

            Convolution2D(8, kernel_size=3, padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(strides=2),

            Dropout(0.3),

            Convolution2D(16, kernel_size=3, padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(strides=2),

            Dropout(0.3),

            Flatten(),
            Dense(256, activation="relu"),
        ])

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        output_a = base_network(input_a)
        output_b = base_network(input_b)

        merged = Concatenate() ([output_a, output_b])
        merged = Dense(128, activation="relu") (merged)
        output = Dense(2, activation="softmax") (merged)

        return Model([input_a, input_b], output)

    def contrastive_loss(y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


    model = buildNetwork(input_shape=input_shape)

    rms = RMSprop()
    sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6)
    model.compile(loss="binary_crossentropy", optimizer=sgd)

    return [model]
