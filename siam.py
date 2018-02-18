from __future__ import division, print_function

import os

import numpy as np
import h5py
from keras.models import Model
from matplotlib import pyplot as plt

from src.model.caps_merge import generate_model
from src.train import train_model
from src.test import cmc, test
from src.generator import trainGenerator, validationGenerator, testGenerator
from src.utils.log import log


def siamRD(model_data,
            b_load_weights=False,
            b_train_model=False,
            b_test_model=False,
            b_no_ui=False):

    # Disable tf debug
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Choose GPU to use
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # Generate model
    model = generate_model(input_shape=model_data["input_shape"])
    log("Generated model")

    # Load weights
    if b_load_weights:
        model.load_weights(os.path.join("weights", model_data["weights_file"]))
        log("Loaded weights")

    # Train
    if b_train_model:

        # Generators
        generator_train = trainGenerator(
                            database=model_data["dataset_path"],
                            batch_size=model_data["batch_size"] )
        generator_val = validationGenerator(
                            database=model_data["dataset_path"],
                            batch_size=model_data["batch_size"] )

        log("Begining training")
        histo = train_model(model,
                            generator_train=generator_train,
                            generator_val=generator_val,
                            batch_size=model_data["batch_size"],
                            steps_per_epoch=model_data["steps_per_epoch"],
                            epochs=model_data["epochs"],
                            validation_steps=model_data["validation_steps"])

    # Test
    if b_test_model:
        generator_test = testGenerator(
                            database=model_data["dataset_path"])

        log("Testing model [cmc]")
        cmc(model, generator_test, b_no_ui)
