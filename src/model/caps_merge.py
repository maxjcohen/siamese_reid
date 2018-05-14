import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Reshape, BatchNormalization, Activation, Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Lambda
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import matplotlib.pyplot as plt
from PIL import Image
from src.model.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

def generate_model(input_shape=(28, 28, 1), lr=0.001):

    def buildNetwork(input_shape=(28, 28, 1)):
        def euclidean_distance(vects):
            x, y = vects
            return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

        n_class = 10
        routings = 3

        # Base network
        x = Input(shape=input_shape)

        conv1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu')(x)

        primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=16, kernel_size=3, strides=2, padding='valid')

        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings)(primarycaps)

        base_network = Model(x, digitcaps)

        # Reid network
        x1 = layers.Input(shape=input_shape)
        x2 = layers.Input(shape=input_shape)

        out1 = base_network(x1)
        out2 = base_network(x2)

        merged = Concatenate() ([out1, out2])
        merged = Reshape((10*32,)) (merged)
        merged = Dropout(0.3) (merged)

        merged = Dense(500, activation="relu") (merged)
        merged = Dropout(0.3) (merged)

        output = Dense(2, activation="softmax") (merged)

        reid_network = Model([x1, x2], output)

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

    def margin_loss(y_true, y_pred):
        L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

        return K.mean(K.sum(L, 1))

    def f1score(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = tp / (pred_pos + K.epsilon())

        real_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = tp / (real_pos + K.epsilon())

        return 2 * precision * recall / (precision + recall + K.epsilon())

    reid_network, feature_network = buildNetwork(input_shape=input_shape)

    # Feature network
    feature_network.compile(optimizer=SGD(lr=0.00001), loss=[margin_loss])

    # Reid network
    opt = SGD(lr=lr)
    reid_network.compile(loss="categorical_crossentropy", optimizer=opt, metrics=[f1score])

    return reid_network, feature_network
