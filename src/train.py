import csv
import time

from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K

from .data_creator import ImageDataGenerator_for_multiinput
from .generator import trainGenerator, validationGenerator

class Logger(Callback):
    def __init__(self, steps_per_epoch=None, batch_size=None):
        # Init logger
        self.csvfile = open("logs/logs.csv", "a")
        fieldnames = ["time", "epoch", "loss (val)", "steps_per_epoch", "batch_size", "lr", "decay"]
        self.writer = csv.DictWriter(self.csvfile, fieldnames=fieldnames)

        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.writer.writeheader()

    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        current_time = time.asctime( time.localtime(time.time()) )
        self.writer.writerow({
                    "time": current_time,
                    "epoch": epoch,
                    "loss (val)": logs.get("val_loss"),
                    "steps_per_epoch": self.steps_per_epoch,
                    "batch_size": self.batch_size,
                    "lr": K.eval(optimizer.lr),
                    "decay": optimizer.initial_decay
        })
        self.csvfile.flush()

    def on_train_end(self, logs={}):
        # Close logger
        self.csvfile.close()

def train_model(model,
                dataset,
                batch_size=32,
                steps_per_epoch=100,
                epochs=10,
                verbose=1,
                validation_steps=10,
                initial_epoch=0):

    # Loggers
    modelCheckpoint = ModelCheckpoint(
                                        filepath="weights/weights_{epoch:02d}_{val_loss:.2f}.hdf5",
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        period=1
    )

    logger = Logger(steps_per_epoch=steps_per_epoch, batch_size=batch_size)

    # Generators
    generator_train = trainGenerator(database=dataset, batch_size=batch_size)
    generator_val = validationGenerator(database=dataset, batch_size=batch_size)


    hist = model.fit_generator(generator=generator_train,
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                verbose=verbose,
                                validation_data=generator_val,
                                validation_steps=validation_steps,
                                initial_epoch=initial_epoch,
                                callbacks=[modelCheckpoint, logger],
    )

    return hist
