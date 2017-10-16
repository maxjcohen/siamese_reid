import argparse
import h5py
import matplotlib.pyplot as plt

from siam import siamRD
from src.model import generate_model, compile_model
from src.train import train_model
from src.test import test_model

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Main Script.')
    parser.add_argument('--dataset',
                        dest='dataset_path',
                        help='HDF5 dataset file path.',
                        required=True)
    args = parser.parse_args()
    return args


siamRD(b_load_weights=False, b_train_model=True, b_test_model=False, verbose=False)
