from __future__ import division, print_function

import os
import threading

from src.model.caps_merge import generate_model
# from src.model.cnn_distance import generate_model
from src.train import train_model
from src.test import cmc, test
from src.generator import ReidGenerator, testGenerator, featureGenerator
from src.utils.log import log

class ReID:
    """docstring for ReID."""
    def __init__(self, model_data, b_no_ui=False):
        self.model_data = model_data

        self.reid_network = None

        self.feature_network = None
        self.pretrain = False

        self.b_no_ui = b_no_ui

        self.__initEnv()
        self.__parseData()


    def run(self,
            b_load_weights=False,
            b_train_model=False,
            b_test_model=False):

        # Generate model
        self.generateNetwork()

        # Load weights
        if b_load_weights:
            self.loadWeights()

        # Train
        if b_train_model:
            self.train()

        # Test
        if b_test_model:
            self.test()


    def __initEnv(self):
        # Disable tf debug
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Choose GPU to use
        # os.environ["CUDA_VISIBLE_DEVICES"]="0"

    def __parseData(self):
        self.input_shape = self.model_data["input_shape"]
        self.dataset = self.model_data["dataset_path"]
        self.batch_size = self.model_data["batch_size"]
        self.steps_per_epoch = self.model_data["steps_per_epoch"]
        self.epochs = self.model_data["epochs"]
        self.validation_steps = self.model_data["validation_steps"]
        self.weights_file = self.model_data["weights_file"]

    def generateNetwork(self):
        networks = generate_model(input_shape=self.input_shape)

        if len(networks) == 2:
            log("Detected 2 networks, will pretrain", "info")
            self.reid_network, self.feature_network = networks
            self.pretrain = True;
        elif len(networks) == 1:
            log("Detected 1 network", "info")
            self.reid_network = networks[0]
        else:
            log("Detected {} networks, aborting".format(len(networks)), "error")
            exit(101)
        log("Model generated")

    def train(self):

        if self.pretrain:
            # Generators
            generator_train = featureGenerator(
                                database=self.dataset,
                                batch_size=self.batch_size,
                                flag="train" )
            generator_val = featureGenerator(
                                database=self.dataset,
                                batch_size=self.batch_size,
                                flag="validation" )

            log("Begining training [features]")
            histo = train_model(self.feature_network,
                                generator_train=generator_train,
                                generator_val=generator_train,
                                batch_size=self.batch_size,
                                steps_per_epoch=self.steps_per_epoch,
                                epochs=self.epochs,
                                validation_steps=self.validation_steps)


        # Generators
        generator_train = ReidGenerator(
                            database=self.dataset,
                            batch_size=self.batch_size,
                            flag="train")
        generator_val = ReidGenerator(
                            database=self.dataset,
                            batch_size=self.batch_size,
                            flag="validation")
        
        log("Begining training [reid]")
        histo = train_model(self.reid_network,
                            generator_train=generator_train,
                            generator_val=generator_train,
                            batch_size=self.batch_size,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            validation_steps=self.validation_steps)

    def test(self):
        generator_test = testGenerator(database=self.dataset)

        log("Testing model [cmc]")
        cmc(self.reid_network, generator_test, self.b_no_ui)

    def loadWeights(self):
        self.reid_network.load_weights(os.path.join("weights", self.weights_file))
        log("Loaded weights")

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
