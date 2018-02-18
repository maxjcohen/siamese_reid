import argparse
import os.path
import sys
import json

import h5py
import matplotlib.pyplot as plt

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Main Script.')
    parser.add_argument('--load_weights', action='store_true', help='Loads weights.')
    parser.add_argument('--train', action='store_true', help='Trains the network.')
    parser.add_argument('--test', action='store_true', help='Test the network.')
    parser.add_argument('--no_ui', action='store_true', help='No graphic interface.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Read parameters from json
    with open("model_parameters.json", "r") as f:
        model_data = json.loads(f.read())

    # Check db exists
    if not os.path.isfile(model_data["dataset_path"]):
        print("\033[1m\033[91mERROR\033[0m dataset {} not found.".format(model_data["dataset_path"]))
        sys.exit(1)

    import siam
    siam.siamRD(
        model_data=model_data,
        b_load_weights=args.load_weights,
        b_train_model=args.train,
        b_test_model=args.test,
        b_no_ui=args.no_ui)
