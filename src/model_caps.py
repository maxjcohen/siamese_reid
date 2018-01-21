import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Reshape, BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Lambda
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import matplotlib.pyplot as plt
from PIL import Image
from src.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

def generate_model():
    def margin_loss(y_true, y_pred):
        L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

        return K.mean(K.sum(L, 1))

    def buildNetwork(input_shape=(28, 28, 1)):
        def euclidean_distance(vects):
            x, y = vects
            return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

        n_class = 10
        routings = 3

        x = layers.Input(shape=input_shape)

        # Layer 1: Just a conventional Conv2D layer
        conv1 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')(x)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=16, kernel_size=9, strides=2, padding='valid')

        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                                 name='digitcaps')(primarycaps)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        # If using tensorflow, this will not be necessary. :)
        out_caps = Length(name='capsnet')(digitcaps)

        base_model = models.Model(x, out_caps)

        base_model = Sequential([
            Convolution2D(4, kernel_size=3, padding="same", input_shape=input_shape),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(strides=2),

            Dropout(0.1),

            Flatten(),
            Dense(128),
        ])


        x1 = layers.Input(shape=input_shape)
        x2 = layers.Input(shape=input_shape)

        out1 = base_model(x1)
        out2 = base_model(x2)

        distance = Lambda(euclidean_distance)([out1, out2])

        return models.Model([x1, x2], distance)

    def contrastive_loss(y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


    model = buildNetwork(input_shape=(160, 60, 3))

    rms = RMSprop()
    sgd = SGD(lr=0.001, momentum=0.9, decay=1e-6)
    # model.compile(loss=contrastive_loss, optimizer=rms)
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss=[margin_loss])

    return model
