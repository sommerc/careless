import os
import n2v
import z5py
import json
import numpy
import argparse

from skimage.transform import resize
from matplotlib import pyplot as plt


def get_args():

    description = """CAREless Noise2Void: command line script for predicting new BS-BDV N5 images"""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("model_path", nargs=1, type=str)
    parser.add_argument(
        "n5_file", type=str, nargs=1, help="N5 File to process with Noise2Void"
    )

    parser.add_argument("--ntiles", nargs=3, type=int, default=[8, 8, 8])
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("-i", "--invert", action="store_true")

    parser.add_argument(
        "-s",
        "--setup_ids",
        type=int,
        nargs="+",
        help="List of Setup ids to predict",
        required=True,
    )

    parser.add_argument("-o", "--out_path", type=str, default=None)

    args = parser.parse_args()
    return args


def copy_attrs(gin, gout):
    in_attrs = gin.attrs
    out_attrs = gout.attrs
    for key, val in in_attrs.items():
        out_attrs[key] = val


def run(args):
    """
    """
    import numpy as np
    from n2v.models import N2VConfig, N2V
    from csbdeep.utils import plot_history
    from n2v.utils.n2v_utils import manipulate_val_data
    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

    model_path = args.model_path[0]
    fn_n5 = args.n5_file[0]

    n_tiles = tuple(args.ntiles)
    n_threads = args.n_threads

    assert os.path.exists(model_path), f"Model path '{model_path}' not found"
    assert os.path.exists(fn_n5), f"N5 file path '{fn_n5}' not found"
    assert fn_n5.endswith(".n5"), "Not a n5 file"

    model_name = os.path.basename(model_path)
    base_dir = os.path.dirname(model_path)

    # We are now creating our network model.
    model = N2V(config=None, name=model_name, basedir=base_dir)

    setup_ids = args.setup_ids

    fh_in = z5py.File(fn_n5, "r")

    out_n5_path = fn_n5[:-3] + "_n2v.n5"

    if args.out_path is not None:
        out_n5_path = os.path.join(
            args.out_path, os.path.basename(fn_n5)[:-3] + "_n2v.n5"
        )

    print(f"Predict into '{out_n5_path}'")

    fh_out = z5py.File(out_n5_path, "a", use_zarr_format=False)

    for sid in setup_ids:
        img = fh_in[f"setup{sid}/timepoint0/s0"]

        img_np = img[:]
        pred_s0 = model.predict(img_np, axes="ZYX"[-img.ndim :], n_tiles=n_tiles)
        pred_s0 = (pred_s0 + 0.5).astype(img.dtype)

        if args.invert:
            print("  - Invert")
            pred_s0 = numpy.iinfo(pred_s0.dtype).max - pred_s0

        print("  - Writing levels s0")

        ds_pred_s0 = fh_out.create_dataset(
            f"setup{sid}/timepoint0/s0",
            shape=img.shape,
            chunks=img.chunks,
            dtype=img.dtype,
            compression=img.compression,
            n_threads=n_threads,
        )

        ds_pred_s0[:] = pred_s0

        copy_attrs(fh_in[f"setup{sid}"], fh_out[f"setup{sid}"])

        # create pyramid

        levels = sorted(fh_in[f"setup{sid}/timepoint0"].keys())[1:]

        ds_down_scaled = ds_pred_s0
        for l in levels:
            ds_orig_sl = fh_in[f"setup{sid}/timepoint0/{l}"]

            ds_down_scaled = resize(
                ds_down_scaled, ds_orig_sl.shape, order=1, preserve_range=True
            )

            print(f"  - Writing levels {l}")
            ds_pred_sl = fh_out.create_dataset(
                f"setup{sid}/timepoint0/{l}",
                shape=ds_orig_sl.shape,
                chunks=ds_orig_sl.chunks,
                dtype=ds_orig_sl.dtype,
                compression=ds_orig_sl.compression,
            )

            ds_pred_sl[:] = ds_down_scaled.astype(ds_orig_sl.dtype)

        copy_attrs(fh_in[f"setup{sid}/timepoint0"], fh_out[f"setup{sid}/timepoint0"])


if __name__ == "__main__":
    args = get_args()
    print("\nN2V predict on BS-BDV N5 file")
    print("#" * 80)
    for arg in vars(args):
        print(f" {arg:28s}", getattr(args, arg))

    run(args)
