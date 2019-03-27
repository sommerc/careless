import os
import sys
import glob
import argparse

description = """BIF-Care command line script for predicting new images given existing project."""

def get_args():
    """
    Helper function for the argument parser.
    """
    parser = argparse.ArgumentParser(description=description)

    # Add arguments
    parser.add_argument('-p', '-project', type=str, action='store', required=True, help="BIF-Care project file")
    parser.add_argument('-t', '-tiles'  ,  type=str, action='store', default="1,4,4", help="Tiles in ZYX (default: 1,4,4)")

    parser.add_argument('files', type=str, nargs='+', help='Images to predict')
    
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    
    assert os.path.exists(args.p), f"Project file {args.p} does not exists"

    n_tiles = args.t.split(",")
    assert len(n_tiles) == 3, f"Tiles parameter not correct {args.t}. Provides list of integers in ZYX, e.g. '1,4,5'"

    n_tiles = list(map(int, n_tiles))

    for fn in args.files:
        assert os.path.exists(fn), f"File for prediction {fn} does not exist"

    print("Starting Care...")
    from bif_care import gui
    from bif_care.core import BifCareTrainer
    from bif_care.utils import JVM
    
    print("Load project file")
    gui.params.load(args.p)
    
    for fn in args.files:
        bt = BifCareTrainer(**gui.params)
        bt.predict(fn, n_tiles=n_tiles)

    JVM().shutdown()


if __name__ == "__main__":
    main()
    