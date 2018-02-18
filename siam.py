from __future__ import division, print_function

import os
import threading

from src.model.caps_merge import generate_model
# from src.model.cnn_distance import generate_model
from src.train import train_model
from src.test import cmc, test
from src.generator import trainGenerator, validationGenerator, testGenerator, featureGenerator
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
    networks = generate_model(input_shape=model_data["input_shape"])

    if len(networks) == 2:
        log("Detected 2 networks, will pretrain", "info")
        reid_network, feature_network = networks
        pretrain = True;
    elif len(networks) == 1:
        log("Detected 1 network", "info")
        reid_network = networks[0]
        feature_network = None
        pretrain = False
    else:
        log("Detected {} networks, aborting".format(len(networks)), "error")
        exit(101)
    log("Model generated")

    # Load weights
    if b_load_weights:
        model.load_weights(os.path.join("weights", model_data["weights_file"]))
        log("Loaded weights")

    # Train
    if b_train_model:

        if pretrain:
            # Generators
            generator_train = featureGenerator(
                                database=model_data["dataset_path"],
                                batch_size=model_data["batch_size"],
                                flag="train" )
            generator_val = featureGenerator(
                                database=model_data["dataset_path"],
                                batch_size=model_data["batch_size"],
                                flag="validation" )

            log("Begining training [features]")
            histo = train_model(feature_network,
                                generator_train=generator_train,
                                generator_val=generator_train,
                                batch_size=model_data["batch_size"],
                                steps_per_epoch=model_data["steps_per_epoch"],
                                epochs=model_data["epochs"],
                                validation_steps=model_data["validation_steps"])


        # Generators
        generator_train = trainGenerator(
                            database=model_data["dataset_path"],
                            batch_size=model_data["batch_size"] )
        generator_val = validationGenerator(
                            database=model_data["dataset_path"],
                            batch_size=model_data["batch_size"] )

        generator_train = threadsafe_iter(generator_train)
        generator_val = threadsafe_iter(generator_val)

        log("Begining training [reid]")
        histo = train_model(reid_network,
                            generator_train=generator_train,
                            generator_val=generator_train,
                            batch_size=model_data["batch_size"],
                            steps_per_epoch=model_data["steps_per_epoch"],
                            epochs=model_data["epochs"],
                            validation_steps=model_data["validation_steps"])


    # Test
    if b_test_model:
        generator_test = testGenerator(
                            database=model_data["dataset_path"])

        log("Testing model [cmc]")
        cmc(reid_network, generator_test, b_no_ui)



class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)
