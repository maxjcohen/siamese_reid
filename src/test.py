import numpy as np
import tqdm

def cmc(model, generator_test, no_ui):
    test_batch_x1, test_batch_x2, n_ids = next(generator_test)

    ranks = np.zeros(n_ids)
    # Compute ranks
    for index, image in tqdm.tqdm(list(enumerate(test_batch_x1))):
        batch_x1 = np.full((n_ids, *image.shape), image)

        netout = model.predict_on_batch([batch_x1, test_batch_x2])
        distances = netout.T[0]
        # Get rank number for this person
        n_rank = np.argwhere(np.argsort(distances) == index)[0, 0]
        ranks[n_rank:] += 1

    ranks = ranks / n_ids

    if not no_ui:
        from matplotlib import pyplot as plt
        # Plot
        plt.plot(ranks)
        for index, value in enumerate(ranks):
            plt.annotate("rg{}: {:.2f}".format(index+1, value), (index, value), xytext=(index+1.7, value-0.02))
        plt.show()

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
