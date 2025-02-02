import os
import sys
import glob
import json
import numpy
import pathlib
import argparse
import tifffile
import pandas as pd
import javabridge as jv
import bioformats as bf
from collections import namedtuple
from matplotlib import pyplot as plt
from skimage.transform import rescale
from tqdm.auto import tqdm

from functools import partial

import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from csbdeep.utils import plot_some
from csbdeep.models import Config, CARE
from csbdeep.io import load_training_data
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.data import RawData, create_patches
from csbdeep.utils import axes_dict, plot_some, plot_history

from .utils import (
    JVM,
    get_file_list,
    get_pixel_dimensions,
    get_upscale_factors,
    get_space_time_resolution,
    Timer,
    check_file_lists,
    save_results,
)


class CareInputConverter(object):
    def __init__(self, **params):
        self.order = 0
        self.__dict__.update(**params)

    def _convert(self, conv_glob, conv_token, conv_scaling=None):
        conversion_files = get_file_list(self.in_dir, conv_glob)
        print("Converting {}".format(conv_token))
        for f_i, f_low in enumerate(tqdm(conversion_files)):
            ir = bf.ImageReader(str(f_low))
            reader = ir.rdr

            loci_pixel_type = reader.getPixelType()

            if loci_pixel_type == 1:
                # uint8
                dtype = numpy.uint8
            elif loci_pixel_type == 3:
                # uint16
                dtype = numpy.uint16
            else:
                print(
                    " -- Error: Pixel-type not supported. Pixel type must be 8- or 16-bit"
                )
                return

            series = 0
            z_size = reader.getSizeZ()
            y_size = reader.getSizeY()
            x_size = reader.getSizeX()
            c_size = reader.getSizeC()

            t_size = reader.getSizeT()

            for t in range(t_size):
                img_3d = numpy.zeros((z_size, c_size, y_size, x_size), dtype=dtype)
                for z in range(z_size):
                    for c in range(c_size):
                        img_3d[z, c, :, :] = ir.read(
                            series=series, z=z, t=t, c=c, rescale=False
                        )

                tmp_dir = pathlib.Path(self.out_dir) / "train_data" / "raw"

                for c in range(c_size):
                    low_dir = tmp_dir / "CH_{}".format(c) / conv_token
                    low_dir.mkdir(parents=True, exist_ok=True)

                    out_tif = low_dir / "training_file_{:04d}_t{:04d}.tif".format(
                        f_i, t
                    )

                    img_3d_ch = img_3d[:, c, :, :]
                    if conv_scaling:
                        img_3d_ch = rescale(
                            img_3d_ch,
                            conv_scaling,
                            preserve_range=True,
                            order=self.order,
                            mode="reflect",
                            anti_aliasing=True,
                        )

                    tifffile.imwrite(
                        out_tif,
                        img_3d_ch[:, None, :, :].astype(dtype),
                        imagej=True,
                        metadata={"axes": "ZCYX"},
                    )
            ir.close()

    def convert(self):
        low_scaling = get_upscale_factors(self.in_dir, self.low_wc, self.high_wc)
        if (numpy.array(low_scaling) == 1).all():
            low_scaling = None
        self._convert(self.low_wc, "low", low_scaling)
        self._convert(self.high_wc, "GT")
        print("Done")


class CareTrainer(object):
    def __init__(self, headless=False, **params):
        self.order = 0
        self.headless = headless

        from .care import params as params_default

        self.__dict__.update(**params_default)
        self.__dict__.update(**params)

    def create_patches(self, normalization=None):
        for ch in self.train_channels:
            n_images = len(
                list(
                    (
                        pathlib.Path(self.out_dir)
                        / "train_data"
                        / "raw"
                        / "CH_{}".format(ch)
                        / "GT"
                    ).glob("*.tif")
                )
            )
            print(
                "-- Creating {} patches for channel: {}".format(
                    n_images * self.n_patches_per_image, ch
                )
            )
            raw_data = RawData.from_folder(
                basepath=pathlib.Path(self.out_dir)
                / "train_data"
                / "raw"
                / "CH_{}".format(ch),
                source_dirs=["low"],
                target_dir="GT",
                axes=self.axes,
            )

            if normalization is not None:
                X, Y, XY_axes = create_patches(
                    raw_data=raw_data,
                    patch_size=self.patch_size,
                    n_patches_per_image=self.n_patches_per_image,
                    save_file=self.get_training_patch_path()
                    / "CH_{}_training_patches.npz".format(ch),
                    verbose=False,
                    normalization=normalization,
                )
            else:
                X, Y, XY_axes = create_patches(
                    raw_data=raw_data,
                    patch_size=self.patch_size,
                    n_patches_per_image=self.n_patches_per_image,
                    save_file=self.get_training_patch_path()
                    / "CH_{}_training_patches.npz".format(ch),
                    verbose=False,
                )

            if not self.headless:
                plt.figure(figsize=(16, 4))

                rand_sel = numpy.random.randint(low=0, high=len(X), size=6)
                plot_some(
                    X[rand_sel, 0], Y[rand_sel, 0], title_list=[range(6)], cmap="gray"
                )

                plt.show()

        print("Done")
        return

    def get_training_patch_path(self):
        return pathlib.Path(self.out_dir) / "train_data" / "patches"

    def train(self, channels=None, **config_args):
        # limit_gpu_memory(fraction=1)
        if channels is None:
            channels = self.train_channels

        with Timer("Training"):
            for ch in channels:
                print("-- Training channel {}...".format(ch))
                (X, Y), (X_val, Y_val), axes = load_training_data(
                    self.get_training_patch_path()
                    / "CH_{}_training_patches.npz".format(ch),
                    validation_split=0.1,
                    verbose=False,
                )

                c = axes_dict(axes)["C"]
                n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

                config = Config(
                    axes,
                    n_channel_in,
                    n_channel_out,
                    train_epochs=self.train_epochs,
                    train_steps_per_epoch=self.train_steps_per_epoch,
                    train_batch_size=self.train_batch_size,
                    probabilistic=self.probabilistic,
                    **config_args,
                )
                # Training

                # if (
                #     pathlib.Path(self.out_dir) / "models" / "CH_{}_model".format(ch)
                # ).exists():
                #     print("config there already")
                #     config = None

                model = CARE(
                    config,
                    "CH_{}_model".format(ch),
                    basedir=pathlib.Path(self.out_dir) / "models",
                )

                # Show learning curve and example validation results
                try:
                    history = model.train(X, Y, validation_data=(X_val, Y_val))
                except tf.errors.ResourceExhaustedError:
                    print(
                        " >> ResourceExhaustedError: Aborting...\n  Training data too big for GPU. Are other GPU jobs running? Perhaps, reduce batch-size or patch-size?"
                    )
                    return
                except tf.errors.UnknownError:
                    print(
                        " >> UnknownError: Aborting...\n  No enough memory available on GPU... are other GPU jobs running?"
                    )
                    return

                hist_df = pd.DataFrame(history.history)
                hist_df.to_csv(f"{self.out_dir}/models/CH_{ch}_training_log.csv")

                if not self.headless:
                    plt.figure(figsize=(16, 5))
                    plot_history(
                        history,
                        ["loss", "val_loss"],
                        ["mse", "val_mse", "mae", "val_mae"],
                    )

                    plt.figure(figsize=(12, 7))
                    _P = model.keras_model.predict(X_val[:5])

                    if self.probabilistic:
                        _P = _P[..., 0]

                    plot_some(X_val[:5], Y_val[:5], _P, pmax=99.5, cmap="gray")
                    plt.suptitle(
                        "5 example validation patches\n"
                        "top row: input (source),  "
                        "middle row: target (ground truth),  "
                        "bottom row: predicted from source"
                    )

                    plt.show()

                print("-- Export model for use in Fiji...")
                model.export_TF()
                print("Done")

    def predict_multiple(self, file_fns, n_tiles=(1, 4, 4)):
        file_list = file_fns.split("\n")
        for file_fn in file_list:
            self.predict(file_fn.strip(), n_tiles=n_tiles)

    def predict(self, file_fn, n_tiles=(1, 4, 4)):
        JVM().start()

        pixel_reso = get_space_time_resolution(file_fn)
        print("Prediction {}".format(file_fn))
        print(" -- Using pixel sizes and frame interval", pixel_reso)

        ir = bf.ImageReader(file_fn)
        reader = ir.rdr

        loci_pixel_type = reader.getPixelType()

        if loci_pixel_type == 1:
            # uint8
            dtype = numpy.uint8
        elif loci_pixel_type == 3:
            # uint16
            dtype = numpy.uint16
        else:
            print("Error: Pixel-type not supported. Pixel type must be 8- or 16-bit")
            return

        if self.probabilistic:
            dtype = numpy.float32

        series = 0
        z_size = reader.getSizeZ()
        y_size = reader.getSizeY()
        x_size = reader.getSizeX()
        c_size = reader.getSizeC()
        t_size = reader.getSizeT()

        z_out_size = int(numpy.round(z_size * self.low_scaling[0]))
        y_out_size = int(numpy.round(y_size * self.low_scaling[1]))
        x_out_size = int(numpy.round(x_size * self.low_scaling[2]))

        if c_size != len(self.train_channels):
            print(
                " -- Warning: Number of Channels during training and prediction do not match. Using channels {} for prediction".format(
                    self.train_channels
                )
            )

        for ch in self.train_channels:
            model = CARE(
                None,
                "CH_{}_model".format(ch),
                basedir=pathlib.Path(self.out_dir) / "models",
            )

            out_channels = 1
            if self.probabilistic:
                out_channels = 2

            res_image_ch = numpy.zeros(
                shape=(t_size, z_out_size, out_channels, y_out_size, x_out_size),
                dtype=dtype,
            )

            print(" -- Predicting channel {}".format(ch))
            for t in tqdm(range(t_size), total=t_size):
                img_3d = numpy.zeros((z_size, y_size, x_size), dtype=dtype)
                for z in range(z_size):
                    img_3d[z, :, :] = ir.read(
                        series=series, z=z, c=ch, t=t, rescale=False
                    )

                img_3d_ch_ex = rescale(
                    img_3d,
                    self.low_scaling,
                    preserve_range=True,
                    order=self.order,
                    mode="reflect",
                    anti_aliasing=True,
                )

                if not self.probabilistic:
                    # non-probabilistic
                    pred = model.predict(img_3d_ch_ex, axes="ZYX", n_tiles=n_tiles)
                    di = numpy.iinfo(dtype)
                    pred = pred.clip(di.min, di.max).astype(dtype)
                    res_image_ch[t, :, 0, :, :] = pred
                else:
                    # probabilistic
                    pred = model.predict_probabilistic(
                        img_3d_ch_ex, axes="ZYX", n_tiles=n_tiles
                    )
                    di = numpy.float32

                    res_image_ch[t, :, 0, :, :] = pred.mean()
                    res_image_ch[t, :, 1, :, :] = pred.scale()

            ch_out_fn = os.path.join(
                os.path.dirname(file_fn),
                os.path.splitext(os.path.basename(file_fn))[0]
                + "_care_predict_ch{}".format(ch),
            )
            print(
                " -- Saving channel {} CARE prediction to file '{}'".format(
                    ch, ch_out_fn
                )
            )

            reso = (
                1 / (pixel_reso.X / self.low_scaling[2]),
                1 / (pixel_reso.Y / self.low_scaling[1]),
            )
            spacing = pixel_reso.Z / self.low_scaling[0]
            unit = pixel_reso.Xunit
            finterval = pixel_reso.T

            save_results(
                ch_out_fn,
                res_image_ch,
                reso,
                finterval,
                spacing,
                unit,
                ome_zarr=self.output_zarr,
            )

            res_image_ch = None  # should trigger gc and free the memory


def get_args():
    description = """CAREless care: command line script for predicting new images given existing project."""

    parser = argparse.ArgumentParser(description=description)

    # Add arguments
    parser.add_argument(
        "--care_project",
        type=str,
        nargs=1,
        help="CAREless care project file (.json)",
        required=True,
    )
    parser.add_argument("files", type=str, nargs="+", help="Files to predict")
    parser.add_argument("--ntiles", nargs=3, type=int, default=[1, 8, 8])

    args = parser.parse_args()

    return args


def cmd_line_predict():
    def get_args():
        description = """CAREless Care: Command line script for predicting new images given a existing project (.json) """

        parser = argparse.ArgumentParser(description=description)

        # Add arguments
        parser.add_argument(
            "--care_project",
            type=str,
            nargs=1,
            help="CAREless care project file (.json)",
            required=True,
        )
        parser.add_argument("files", type=str, nargs="+", help="Files to predict")
        parser.add_argument("--ntiles", nargs=3, type=int, default=[1, 8, 8])

        args = parser.parse_args()

        return args

    from .care import GuiParams

    args = get_args()
    assert os.path.exists(args.care_project[0]), (
        f"Project file '{args.care_project[0]}'' does not exist."
    )

    params = GuiParams()
    params.load(args.care_project[0])

    print("\n\nCAREless care parameters")
    print("-" * 50)
    for k, v in params.items():
        print("{:25s}: {}".format(k, v))
    print("n-tiles:", args.ntiles)

    print("\n")

    for fn in args.files:
        assert os.path.exists(fn), f"File for prediction {fn} does not exist"

    print("Predicting Careless...")

    bt = CareTrainer(**params)
    for fn in args.files:
        bt.predict(fn, n_tiles=args.ntiles)

    JVM().shutdown()


def cmd_line_train():
    def get_args():
        description = """CAREless Care: Command line script for training given a existing project (.json) """

        parser = argparse.ArgumentParser(description=description)

        # Add arguments
        parser.add_argument(
            "--care_project",
            type=str,
            nargs=1,
            help="CAREless care project file (.json)",
            required=True,
        )

        args = parser.parse_args()

        return args

    from .care import GuiParams

    args = get_args()
    assert os.path.exists(args.care_project[0]), (
        f"Project file '{args.care_project[0]}'' does not exist."
    )

    params = GuiParams()
    params.load(args.care_project[0])

    print("\n\nCAREless care parameters")
    print("-" * 50)
    for k, v in params.items():
        print("{:25s}: {}".format(k, v))

    print("\n")

    check_ok, msg = check_file_lists(
        params["in_dir"], params["low_wc"], params["high_wc"]
    )
    if not check_ok:
        print(msg)
        return

    params["low_scaling"] = get_upscale_factors(
        params["in_dir"], params["low_wc"], params["high_wc"]
    )

    z_dim = get_pixel_dimensions(get_file_list(params["in_dir"], params["low_wc"])[0]).z

    if z_dim == 1:
        params["axes"] = "YX"
    else:
        params["axes"] = "ZYX"

    print("Using CARE in {}D mode".format(len(params["axes"])))

    if (numpy.array(params["low_scaling"]) == 1).all():
        print("Low quality images match high quality resolution")
    else:
        print(
            "Low quality images are scaled up by ({}, {}, {}) in ZYX to match high quality resolution".format(
                *params["low_scaling"]
            )
        )

    CareInputConverter(
        in_dir=params["in_dir"],
        out_dir=params["out_dir"],
        low_wc=params["low_wc"],
        high_wc=params["high_wc"],
    ).convert()

    params.save()
    trainer = CareTrainer(headless=True, **params)
    trainer.create_patches()
    trainer.train()

    # for fn in args.files:
    #     assert os.path.exists(fn), f"File for prediction {fn} does not exist"

    # print("Predicting Careless...")

    # bt = CareTrainer(**params)
    # for fn in args.files:
    #     bt.predict(fn, n_tiles=args.ntiles)

    JVM().shutdown()
