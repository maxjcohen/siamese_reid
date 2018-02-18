
import random

import numpy as np
from keras.datasets import mnist
import h5py

# Load mnist from keras [overkill]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
input_shape = (*x_train.shape[1:], 1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Create h5 file
db = h5py.File("mnist.h5", "w")

groups = {}
groups["train"] = db.create_group("train")
groups["validation"] = db.create_group("validation")

for i in range(10):
    ar = x_train[y_train==i]
    groups["train"].create_dataset(str(i), data=ar)

for i in range(10):
    ar = x_test[y_test==i]
    groups["validation"].create_dataset(str(i), data=ar)

db.close()
