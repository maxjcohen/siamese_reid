import numpy as np
import h5py
import tqdm

def convertCuhk(database="cuhk-03.h5", output="cuhk.h5"):
    with h5py.File(database, "r") as f:
        with h5py.File(output, "w") as db:
            groups = {}
            groups["test"] = db.create_group("test")
            groups["train"] = db.create_group("train")
            groups["validation"] = db.create_group("validation")

            for group in groups:
                for i in tqdm.tqdm(range(len(f["a"][group]))):
                    n_a = f["a"][group][str(i)].shape[0]
                    n_b = f["b"][group][str(i)].shape[0]

                    ar = np.zeros((n_a+n_b, 160, 60, 3))

                    if n_a == 0:
                        ar = f["b"][group][str(i)].value
                    else:
                        ar[:n_a] = f["a"][group][str(i)].value
                        ar[n_a:] = f["b"][group][str(i)].value


                    groups[group].create_dataset(str(i), data=ar)

if __name__ == '__main__':
    convertCuhk()
