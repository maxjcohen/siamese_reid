from __future__ import division, print_function

import os
import json

import numpy as np
import cv2
import h5py
from keras.models import Model
from matplotlib import pyplot as plt

from src.model import generate_model, compile_model
from src.train import train_model
from src.test import test_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def siamBigBro(b_gen_model=False, **goliath_args):
    if b_gen_model:
        model = generate_model()
        model = compile_model(model)
        return model

    return goliath(**goliath_args)


def siamRD(model_parameters_path="model_parameters.json",
            b_load_weights=False,
            b_train_model=False,
            b_test_model=False,
            verbose=True):

    if not verbose:
        # avoid printing TF debugging information
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Read infos from json
    if verbose:
        print("Reading infos from {} .".format(model_parameters_path))

    with open(model_parameters_path, "r") as f:
        model_data = json.loads(f.read())

    # Generate model
    model = generate_model()
    model = compile_model(model)
    if verbose:
        model.summary()

    # Load weights
    if b_load_weights:
        model.load_weights(os.path.join("weights", model_data["weights_file"]))
        if verbose:
            print("Loaded weights/{} .".format(model_data["weights_file"]))

    # Train
    if b_train_model:
        batch_size = model_data["batch_size"]
        batch_per_epoch = model_data["batch_per_epoch"]
        epochs = model_data["epochs"]
        batch_per_valid = model_data["batch_per_valid"]
        # with h5py.File(model_data["dataset_path"], "r") as f:
        #    histo = train_model(model, f, batch_size=batch_size, steps_per_epoch=batch_per_epoch, epochs=epochs, validation_steps=batch_per_valid)
        print("training")

    # Test
    if b_test_model:
        input_folder = model_data["input_folder"]
        output_folder = model_data["output_folder"]
        test_model(model)

if __name__ == '__main__':
    siamRD()
