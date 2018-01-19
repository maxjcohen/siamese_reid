from __future__ import division, print_function

import os

import numpy as np
import h5py
from keras.models import Model
from matplotlib import pyplot as plt

from src.model import generate_model
from src.train import train_model
from src.test import cmc

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def siamRD(model_data,
            b_load_weights=False,
            b_train_model=False,
            b_test_model=False,
            b_no_ui=False):


    # Generate model
    model = generate_model()

    # Load weights
    if b_load_weights:
        model.load_weights(os.path.join("weights", model_data["weights_file"]))

    # Train
    if b_train_model:
        batch_size = model_data["batch_size"]
        steps_per_epoch = model_data["steps_per_epoch"]
        epochs = model_data["epochs"]
        validation_steps = model_data["validation_steps"]

        histo = train_model(model, dataset=model_data["dataset_path"], batch_size=batch_size, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_steps=validation_steps)

    # Test
    if b_test_model:
        cmc(model, b_no_ui)
