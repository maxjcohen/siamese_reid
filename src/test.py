import os
import shutil

# import cv2
from PIL import Image
import numpy as np
import h5py
from matplotlib import pyplot as plt

from src.model import generate_model
from src.goliath import goliath

module_path = os.path.dirname(os.path.abspath(__file__))
wp4_path = os.path.join(module_path, '../..')


def test_model(model,
               verbose=True):
    print("Testing of goliath on cuhk03 dataset.")

    input_folder = os.path.join(wp4_path, 'dataset/cuhk03/input')
    output_folder = os.path.join(wp4_path, 'dataset/cuhk03/output')

    assert os.path.isdir(input_folder)
    assert os.path.isdir(output_folder)

    input_images = sorted([file for file in os.listdir(input_folder)
                                if os.path.isfile(os.path.join(input_folder, file))
                                and file.endswith(".jpg")])

    for image_path in input_images:
        query = np.array(Image.open(os.path.join(input_folder, image_path)))/255
        query_id, score_max = goliath(model, query, output_folder, verbose=verbose)

        if query_id == -1:
            query_id = len(os.listdir(output_folder))
        if not os.path.isdir( os.path.join(output_folder, "id_{}".format(query_id)) ):
            os.makedirs(os.path.join(output_folder, "id_{}".format(query_id)))

        shutil.copyfile(os.path.join(input_folder, image_path),
                        os.path.join(output_folder, "id_{}".format(query_id), os.path.basename(image_path)))


# At final phase, we will not need this function any more.
# This function allows reading the dataset in the .h5 format where the full dataset
# is already available (images).
def _get_test_data_new():
    fp = os.path.join(wp4_path, 'dataset/cuhk03')

    n_ids = 100
    cameraA = np.zeros((n_ids, 160, 60, 3))
    cameraB = np.zeros((n_ids, 160, 60, 3))

    for index_pair in range(n_ids):
        image_x1 = cv2.imread( os.path.join(fp, "a/test", str(index_pair), "0.jpg") )
        image_x2 = cv2.imread( os.path.join(fp, "b/test", str(index_pair), "0.jpg") )

        image_x1 = cv2.cvtColor(image_x1, cv2.COLOR_BGR2RGB) / 255.
        image_x2 = cv2.cvtColor(image_x2, cv2.COLOR_BGR2RGB) / 255.

        cameraA[index_pair] = image_x1
        cameraB[index_pair] = image_x2
    return cameraA, cameraB

def _get_test_data(h5pyfile='cuhk-03.h5', val_or_test='test'):
    h5pyfile = "cuhk03.h5"
    with h5py.File(h5pyfile, 'r') as ff:
        a = np.array([ff['a'][val_or_test][str(i)][0] for i in range(100)])
        b = np.array([ff['b'][val_or_test][str(i)][0] for i in range(100)])

    return a, b

def cmc(model, val_or_test='test'):

        a, b = _get_test_data()

        def _cmc_curve(model,
                       camera1,
                       camera2,
                       rank_max=50):
            num = camera1.shape[0]
            rank = []
            score = []
            camera_batch1 = np.zeros(camera1.shape)
            for i in range(num):
                print("Current id: {}\r".format(i), end="")
                for j in range(num):
                    camera_batch1[j] = camera1[i]
                similarity_batch = model.predict_on_batch([camera_batch1, camera2])
                sim_trans = similarity_batch.transpose()
                similarity_rate_sorted = np.argsort(sim_trans[0])
                for k in range(num):
                    if similarity_rate_sorted[k] == i:
                        rank.append(k+1)
                        break
            rank_val = 0
            for i in range(rank_max):
                rank_val = rank_val + len([j for j in rank if i == j-1])
                score.append(rank_val / float(num))
            return np.array(score)
            print()

        # cmc_curve = _cmc_curve(model, a, b)
        cmc_curve = _cmc_curve(model, a, b)
        print(cmc_curve)

        # Plot
        x_cmc_curve = np.arange(1, len(cmc_curve)+1)
        plt.plot(x_cmc_curve, cmc_curve, 'bo-')
        for index, value in enumerate(cmc_curve):
            plt.annotate("rg {}: {:.2f}".format(index+1, value), (index, value), xytext=(index+1.7, value-0.02))
        plt.show()
        return cmc_curve

if __name__ == "__main__":
    print("Generate model...")
    model = generate_model()
    model = compile_model(model)
    print("Load weights...")
    model.load_weights(os.path.join(wp4_path, 'siamcvpr2015/weights/cuhk03_1.h5'))
    test_model(model)
