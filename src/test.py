import numpy as np
import h5py
from matplotlib import pyplot as plt

def cmc(model, no_ui):
    test_batch, n_ids = _getTestData()

    ranks = np.zeros(n_ids)
    # Compute ranks
    for index, image in enumerate(test_batch):
        print("Current id: {}\r".format(index), end="")

        batch_x1 = np.full((n_ids, 160, 60, 3), image)

        netout = model.predict_on_batch([batch_x1, test_batch])
        distances = netout.T[0]

        # Get rank number for this person
        n_rank = np.argwhere(np.argsort(distances) == index)[0, 0]
        ranks[n_rank:] += 1
    print()

    if not no_ui:
        # Plot
        ranks = ranks / n_ids
        plt.plot(ranks)
        for index, value in enumerate(ranks):
            plt.annotate("rg{}: {:.2f}".format(index+1, value), (index, value), xytext=(index+1.7, value-0.02))
        plt.show()

    return ranks

def _getTestData(database="cuhk.h5"):
    # Open database
    with h5py.File(database, "r") as db:
        n_ids = len(db["test"])

        test_batch = np.zeros((n_ids, 160, 60, 3))

        for index in range(n_ids):
            test_batch[index] = db["test"][str(index)][0]

    return test_batch, n_ids
