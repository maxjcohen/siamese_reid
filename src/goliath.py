import os

from PIL import Image
import numpy as np

def goliath(model,
            query,
            database_path,
            threshold=0.96,
            use_baseline=False,
            verbose=True):

    def get_score(query, subfolder, model, use_baseline=False):
        scores = []
        id_images = [file for file in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, file)) and file.endswith(".jpg")]
        targets = [np.array(Image.open(os.path.join(subfolder, image_path)))/255 for image_path in id_images]
        for i in range(len(targets)):
            scores.append(model.predict([np.array([targets[i]]), np.array([query])], verbose=0)[0][1])
        return np.average(np.array(scores))

    database_subfolders = [d for d in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, d))]

    if len(database_subfolders) == 0:
        if verbose:
            print("No ID yet, empty database")
        return -1, 0

    scores = [get_score(query, os.path.join(database_path, subf), model) for subf in database_subfolders]
    print("Scores : ")
    print(scores)
    score_max = np.max(scores)

    if score_max > threshold:
        query_id = np.argmax(scores)
        if verbose:
            print("query is id = {} (score = {})".format(query_id, score_max))
    else:
        query_id = -1
        if verbose:
            print("query not found (scrore={})".format(score_max))

    return query_id, score_max
