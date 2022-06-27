import os
import n2v
import json
import z5py
import json
import numpy
import argparse
from matplotlib import pyplot as plt


def string_to_slice(slice_string):
    return slice(
        *[
            {True: lambda n: None, False: int}[x == ""](x)
            for x in (slice_string.split(":") + ["", "", ""])[:3]
        ]
    )


class MockViewForN2VPatchGen:
    """
    Class create a z5py array view for generating patches with N2V without ever
    reading all data. This hack exploits the access pattern in n2v datagenarator

    img[s][np.newaxis]

    to inject a z5array instead of numpy (which would requrie full read) into
    generate_patches().

    tested with N2V 0.3.1

    """

    def __init__(self, z5arr):
        self.z5arr = z5arr

    def __getitem__(self, tup):
        if (not isinstance(tup, (tuple,))) and tup == 0:
            return self.__class__(self.z5arr)

        elif tup is None:
            return self

        elif (len(tup) > 1) and tup[0] == 0:
            # actual access
            return self.z5arr[tup[1:]]

        else:
            raise RuntimeError("Should not happen")

    @property
    def shape(self):
        return (1,) + self.z5arr.shape + (1,)

    def __len__(self):
        return 1


def get_args():

    description = """CAREless Noise2Void: command line script for predicting new images given existing project."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--n2v_project",
        type=str,
        nargs=1,
        help="CAREless Noise2Void project file (.json)",
        required=True,
    )

    parser.add_argument(
        "n5_file", type=str, nargs=1, help="N5 File to process with Noise2Void"
    )

    parser.add_argument(
        "-s",
        "--setup_ids",
        type=str,
        action="store",
        default=":",
        help="Select Setup ids by slicing (default all = ':')",
    )

    args = parser.parse_args()
    return args


def run(args):
    """
    """
    fn_param = args.n2v_project[0]
    with open(fn_param, "rt") as jfh:
        params = json.load(jfh)

    print("-" * 80)
    for k, v in params.items():
        print(f" {k:28s}", v)
    print()

    import numpy as np
    from n2v.models import N2VConfig, N2V
    from csbdeep.utils import plot_history
    from n2v.utils.n2v_utils import manipulate_val_data
    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

    fn_n5 = args.n5_file[0]
    fn_param = args.n2v_project[0]
    setup_ids_str = args.setup_ids
    setup_ids_slc = string_to_slice(setup_ids_str)

    fh = z5py.File(fn_n5, "r")
    setup_ids = list(range(len(fh)))[setup_ids_slc]

    img_gen = (MockViewForN2VPatchGen(fh[f"setup{s}/timepoint0/s0"]) for s in setup_ids)

    patches = N2V_DataGenerator().generate_patches_from_list(
        img_gen,
        num_patches_per_img=params["n_patches_per_image"],
        shape=params["patch_size"],
        augment=False,
    )

    patches = patches[..., None]
    numpy.random.shuffle(patches)

    sep = int(len(patches) * 0.9)
    X = patches[:sep]
    X_val = patches[sep:]

    config = N2VConfig(
        X,
        train_steps_per_epoch=params["train_steps_per_epoch"],
        train_epochs=params["train_epochs"],
        train_loss="mse",
        train_batch_size=params["train_batch_size"],
    )

    model_name = "{}_s{}".format(params["name"], setup_ids_str.replace(":", "-"))

    model = N2V(config=config, name=model_name, basedir=os.path.dirname(fn_n5))

    history = model.train(X, X_val)

    with open(f"{model_name}_history.json", "w") as f:
        json.dump(history.history, f)

    # val_patch = X_val[0, ..., 0]
    # val_patch_pred = model.predict(val_patch, axes=params["axes"])

    # val_patch = val_patch.max(0)
    # val_patch_pred = val_patch_pred.max(0)

    # f, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax[0].imshow(val_patch, cmap="gray")
    # ax[0].set_title("Validation Patch")
    # ax[1].imshow(val_patch_pred, cmap="gray")
    # ax[1].set_title("Validation Patch N2V")

    # plt.figure(figsize=(16, 5))
    # plot_history(history, ["loss", "val_loss"])
    # plt.show()


if __name__ == "__main__":
    args = get_args()
    print("\nN2V train on BS-BDV N5 file")
    print("#" * 80)
    for arg in vars(args):
        print(f" {arg:28s}", getattr(args, arg))

    run(args)

    print("Done")
