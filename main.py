import argparse
import h5py
import matplotlib.pyplot as plt

from siam import siamRD

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Main Script.')
    parser.add_argument('--dataset',
                        dest='dataset_path',
                        help='HDF5 dataset file path.',
                        required=True)
    args = parser.parse_args()
    return args

siamRD(b_load_weights=False, b_train_model=False, b_test_model=False)
