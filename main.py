import argparse
import os.path
import sys
import json

from src.utils.log import log

# TODO: display learning history after training
# TODO: learning curve
# TODO: define metrics
# TODO: except error from loading weights

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
    if not os.path.isfile("model_parameters.json"):
        log("model_parameters.json not found", "error")
        sys.exit(1)
    with open("model_parameters.json", "r") as f:
        model_data = json.loads(f.read())

    # Check db exists
    if not os.path.isfile(model_data["dataset_path"]):
        log("dataset {} not found".format(model_data["dataset_path"]), "error")
        sys.exit(1)

    import siam
    model = siam.ReID(model_data, b_no_ui=args.no_ui)
    model.run(
        b_load_weights=args.load_weights,
        b_train_model=args.train,
        b_test_model=args.test)
