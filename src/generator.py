import numpy as np
import h5py

def trainGenerator(database="cuhk.h5", batch_size=32):
    # Open database
    with h5py.File(database, "r") as db:
        n_ids = len(db["train"])

        while True:
            batch_x = np.zeros((batch_size, 2, 160, 60, 3))
            batch_y = np.zeros((batch_size, 2))

            # Choose positive or negative pair
            pairs = np.random.choice(["positive", "negative"], p=[1/3, 2/3], size=batch_size)

            for index, pair in enumerate(pairs):
                if pair == "positive":
                    pair_id = np.random.choice(n_ids)
                    pair_a, pair_b = np.random.choice(len(db["train"][str(pair_id)]), size=2, replace=False)

                    batch_x[index][0] = db["train"][str(pair_id)][pair_a]
                    batch_x[index][1] = db["train"][str(pair_id)][pair_b]
                    batch_y[index] = [0, 1]

                else:
                    pair_ids = np.random.choice(n_ids, 2, replace=False)
                    pair_a = np.random.choice(len(db["train"][str(pair_ids[0])]))
                    pair_b = np.random.choice(len(db["train"][str(pair_ids[1])]))


                    batch_x[index][0] = db["train"][str(pair_ids[0])][pair_a]
                    batch_x[index][1] = db["train"][str(pair_ids[1])][pair_b]
                    batch_y[index] = [1, 0]

            yield batch_x, batch_y

if __name__ == '__main__':
    trainGenerator()
