import numpy as np
import h5py
from matplotlib import pyplot as plt
import tqdm

def cmc(model, no_ui, database="cuhk.h5"):
    test_batch, n_ids = _getTestData(database)

    ranks = np.zeros(n_ids)
    # Compute ranks
    for index, image in tqdm.tqdm(list(enumerate(test_batch))):
        batch_x1 = np.full((n_ids, *image.shape), image)

        netout = model.predict_on_batch([batch_x1, test_batch])
        distances = netout.T[0]

        # Get rank number for this person
        n_rank = np.argwhere(np.argsort(distances) == index)[0, 0]
        ranks[n_rank:] += 1

    ranks = ranks / n_ids

    if not no_ui:
        # Plot
        plt.plot(ranks)
        for index, value in enumerate(ranks):
            plt.annotate("rg{}: {:.2f}".format(index+1, value), (index, value), xytext=(index+1.7, value-0.02))
        plt.show()
    
    else:
        print(ranks)

    return ranks

def test(gen, model):
    batch = next(gen)
    distances = model.predict_on_batch(batch[0])
    n_images = 10
    fig = plt.figure(figsize=(20, 5))

    for i in range(n_images):
        ax = plt.subplot(2, n_images, i+1)
        plt.imshow(batch[0][0][i])
        ax.set_axis_off()
        # ax.set_title(("Same" if batch[1][i][1] else "Diff"))
        ax.set_title(str(distances[i]))
        ax = plt.subplot(2, n_images, n_images+i+1)
        plt.imshow(batch[0][1][i])
        ax.set_axis_off()
    plt.show()

def _getTestData(database="cuhk.h5"):
    # Open database
    with h5py.File(database, "r") as db:
        n_ids = len(db["test"])
        image_shape = db["validation"]["0"].shape[1:]

        test_batch = np.zeros((n_ids, *image_shape))

        for index in range(n_ids):
            test_batch[index] = db["test"][str(index)][0]

    return test_batch, n_ids
