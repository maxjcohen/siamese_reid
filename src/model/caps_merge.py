import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Reshape, BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Lambda
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import matplotlib.pyplot as plt
from PIL import Image
from src.model.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

def generate_model(input_shape=(28, 28, 1)):
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


        # Base network
        x = layers.Input(shape=input_shape)

        conv1 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')(x)

        primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=16, kernel_size=3, strides=2, padding='valid')

        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                                 name='digitcaps')(primarycaps)

        out_caps = Length(name='capsnet')(digitcaps)

        base_network = models.Model(x, out_caps)

        # Reid network
        x1 = layers.Input(shape=input_shape)
        x2 = layers.Input(shape=input_shape)

        out1 = base_network(x1)
        out2 = base_network(x2)

        distance = Lambda(euclidean_distance)([out1, out2])

        reid_network = models.Model([x1, x2], distance)

        # Decoder
        decoder = Sequential([
            Reshape((16*n_class,), input_shape=(16, n_class)),
            Dense(512, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(np.prod(input_shape), activation='sigmoid'),
            Reshape(target_shape=input_shape)
        ])

        feature_network = Model(x, decoder(digitcaps))

        return reid_network, feature_network

    def contrastive_loss(y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


    reid_network, feature_network = buildNetwork(input_shape=input_shape)

    # Feature network
    feature_network.compile(optimizer=Adam(lr=0.001), loss=[margin_loss])

    # Reid network
    rms = RMSprop()
    reid_network.compile(loss=contrastive_loss, optimizer=rms)

    return reid_network, feature_network
