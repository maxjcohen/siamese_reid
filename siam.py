from __future__ import division, print_function

import os

import numpy as np
import h5py
from keras.models import Model
from matplotlib import pyplot as plt

from src.model_caps import generate_model
from src.train import train_model
from src.test import cmc, test

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def siamRD(model_data,
            b_load_weights=False,
            b_train_model=False,
            b_test_model=False,
            b_no_ui=False):


    # Generate model
    model = generate_model(input_shape=(160, 60, 3))

    # Load weights
    if b_load_weights:
        model.load_weights(os.path.join("weights", model_data["weights_file"]))

    # Train
    if b_train_model:

        histo = train_model(model,
                                dataset=model_data["dataset_path"], batch_size=model_data["batch_size"],
                                steps_per_epoch=model_data["steps_per_epoch"],
                                epochs=model_data["epochs"],
                                validation_steps=model_data["validation_steps"])

    # Test
    if b_test_model:
        cmc(model, b_no_ui, model_data["dataset_path"])
        # from src.generator import trainGenerator
        # gen = trainGenerator(batch_size=32)
        # test(gen, model)
