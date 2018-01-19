import argparse
import os.path
import sys

import h5py
import matplotlib.pyplot as plt

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Main Script.')
    parser.add_argument('--convert', dest='dataset_path', help='convert original dataset to format.')
    parser.add_argument('--load_weights', action='store_true', help='Loads weightself.')
    parser.add_argument('--train', action='store_true', help='Trains the network.')
    parser.add_argument('--test', action='store_true', help='Test the network.')
    parser.add_argument('--no_ui', action='store_true', help='No graphic interface.')
    return parser.parse_args()

# siamRD(b_load_weights=False, b_train_model=False, b_test_model=False)

if __name__ == '__main__':
    args = parse_args()

    # Convert database [cuhk]
    if args.dataset_path is not None:
        if not os.path.isfile(args.dataset_path):
            print("\033[1m\033[91mERROR\033[0m {} not found.".format(args.dataset_path))
            sys.exit(1)
        print("\033[1m\033[94mConverting\033[0m {} [dataset cuhk]".format(args.dataset_path))

        import src.dataset.cuhk as cuhk
        cuhk.convertCuhk(database=args.dataset_path, output="bla.h5")

        print("\033[1m\033[94mDone\033[0m")

    if args.load_weights or args.train or args.test:
        import siam
        siam.siamRD(
            b_load_weights=args.load_weights,
            b_train_model=args.train,
            b_test_model=args.test,
            b_no_ui=args.no_ui)
