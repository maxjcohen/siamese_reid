import numpy as np
import h5py

def trainGenerator(database="cuhk.h5", batch_size=32):
    # Open database
    with h5py.File(database, "r") as db:
        n_ids = len(db["train"])
        image_shape = db["train"]["0"].shape[1:]

        while True:
            batch_x_1 = np.zeros((batch_size, *image_shape))
            batch_x_2 = np.zeros((batch_size, *image_shape))
            batch_y = np.zeros((batch_size, 2))

            # Choose positive or negative pair
            pairs = np.random.choice(["positive", "negative"], p=[1/3, 2/3], size=batch_size)

            for index, pair in enumerate(pairs):
                if pair == "positive":
                    pair_id = np.random.choice(n_ids)
                    pair_a, pair_b = np.random.choice(len(db["train"][str(pair_id)]), size=2, replace=False)

                    batch_x_1[index] = db["train"][str(pair_id)][pair_a]
                    batch_x_2[index] = db["train"][str(pair_id)][pair_b]
                    batch_y[index] = [0, 1]

                else:
                    pair_ids = np.random.choice(n_ids, 2, replace=False)
                    pair_a = np.random.choice(len(db["train"][str(pair_ids[0])]))
                    pair_b = np.random.choice(len(db["train"][str(pair_ids[1])]))


                    batch_x_1[index] = db["train"][str(pair_ids[0])][pair_a]
                    batch_x_2[index] = db["train"][str(pair_ids[1])][pair_b]
                    batch_y[index] = [1, 0]

            yield [batch_x_1, batch_x_2], batch_y

def validationGenerator(database="cuhk.h5", batch_size=32):
    # Open database
    with h5py.File(database, "r") as db:
        n_ids = len(db["validation"])
        image_shape = db["validation"]["0"].shape[1:]

        while True:
            batch_x_1 = np.zeros((batch_size, *image_shape))
            batch_x_2 = np.zeros((batch_size, *image_shape))
            batch_y = np.zeros((batch_size, 2))

            for index in range(batch_size):
                pair_ids = np.random.choice(n_ids, 2, replace=True)
                pair_a = np.random.choice(len(db["validation"][str(pair_ids[0])]))
                pair_b = np.random.choice(len(db["validation"][str(pair_ids[1])]))


                batch_x_1[index] = db["validation"][str(pair_ids[0])][pair_a]
                batch_x_2[index] = db["validation"][str(pair_ids[1])][pair_b]

                label = (pair_a == pair_b)
                batch_y[index] = [(not label), label]

            yield [batch_x_1, batch_x_2], batch_y


if __name__ == '__main__':
    trainGenerator()
