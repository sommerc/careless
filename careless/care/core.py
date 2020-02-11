import os
import sys
import glob
import json
import numpy
import pathlib
import argparse
import tifffile
import javabridge as jv
import bioformats as bf
from collections import namedtuple
from matplotlib import pyplot as plt
from skimage.transform import rescale
from tqdm import tqdm_notebook as tqdm

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from csbdeep.utils import plot_some
from csbdeep.models import Config, CARE
from csbdeep.io import load_training_data
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.data import RawData, create_patches
from csbdeep.utils import axes_dict, plot_some, plot_history

from .utils import JVM, get_file_list, get_pixel_dimensions, \
                   get_upscale_factors, get_space_time_resolution, Timer


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
                print(" -- Error: Pixel-type not supported. Pixel type must be 8- or 16-bit")
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
                        img_3d[z, c, :, :] = ir.read(series=series,
                                                    z=z,
                                                    t=t,
                                                    c=c, rescale=False)

                tmp_dir = pathlib.Path(self.out_dir) / "train_data" / "raw"

                for c in range(c_size):
                    low_dir = tmp_dir / "CH_{}".format(c) / conv_token
                    low_dir.mkdir(parents=True, exist_ok=True)

                    out_tif = low_dir / "training_file_{:04d}_t{:04d}.tif".format(f_i, t)

                    img_3d_ch = img_3d[:, c, :, :]
                    if conv_scaling:
                        img_3d_ch = rescale(img_3d_ch, conv_scaling, preserve_range=True,
                                            order=self.order,
                                            multichannel=False,
                                            mode="reflect",
                                            anti_aliasing=True)

                    tifffile.imsave(out_tif, img_3d_ch[:, None, :, :].astype(dtype),
                                    imagej=True,
                                    metadata={'axes': 'ZCYX'})
            ir.close()


    def convert(self):
        low_scaling = get_upscale_factors(self.in_dir, self.low_wc, self.high_wc)
        if (numpy.array(low_scaling) == 1).all():
            low_scaling = None
        self._convert(self.low_wc, "low", low_scaling)
        self._convert(self.high_wc, "GT")
        print("Done")

class CareTrainer(object):
    def __init__(self, **params):
        self.order = 0

        if len(params) == 0:
            from .care import params
        self.__dict__.update(**params)

    def create_patches(self):
        for ch in self.train_channels:
            n_images = len(list((pathlib.Path(self.out_dir) / "train_data" / "raw" / "CH_{}".format(ch) / "GT").glob("*.tif")))
            print("-- Creating {} patches for channel: {}".format(n_images*self.n_patches_per_image, ch))
            raw_data = RawData.from_folder (
                                    basepath    = pathlib.Path(self.out_dir) / "train_data" / "raw" / "CH_{}".format(ch),
                                    source_dirs = ['low'],
                                    target_dir  = 'GT',
                                    axes        = self.axes,
                                    )

            X, Y, XY_axes = create_patches (
                raw_data            = raw_data,
                patch_size          = self.patch_size,
                n_patches_per_image = self.n_patches_per_image,
                save_file           = self.get_training_patch_path() / 'CH_{}_training_patches.npz'.format(ch),
                verbose             = False,
            )

            plt.figure(figsize=(16,4))

            rand_sel = numpy.random.randint(low=0, high=len(X), size=6)
            plot_some(X[rand_sel, 0],Y[rand_sel, 0],title_list=[range(6)], cmap="gray")

            plt.show()

        print("Done")
        return


    def get_training_patch_path(self):
        return pathlib.Path(self.out_dir) / 'train_data' / 'patches'

    def train(self, channels=None, **config_args):
        #limit_gpu_memory(fraction=1)
        if channels is None:
            channels = self.train_channels

        with Timer('Training'):

            for ch in channels:
                print("-- Training channel {}...".format(ch))
                (X,Y), (X_val,Y_val), axes = load_training_data(self.get_training_patch_path() / 'CH_{}_training_patches.npz'.format(ch), validation_split=0.1, verbose=False)

                c = axes_dict(axes)['C']
                n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

                config = Config(axes, n_channel_in, n_channel_out, train_epochs=self.train_epochs,
                                                                   train_steps_per_epoch=self.train_steps_per_epoch,
                                                                   train_batch_size=self.train_batch_size,
                                                            **config_args,)
                # Training
                model = CARE(config, 'CH_{}_model'.format(ch), basedir=pathlib.Path(self.out_dir) / 'models')

                # Show learning curve and example validation results
                try:
                    history = model.train(X,Y, validation_data=(X_val,Y_val))
                except tf.errors.ResourceExhaustedError:
                    print(" >> ResourceExhaustedError: Aborting...\n  Training data too big for GPU. Are other GPU jobs running? Perhaps, reduce batch-size or patch-size?")
                    return
                except tf.errors.UnknownError:
                    print(" >> UnknownError: Aborting...\n  No enough memory available on GPU... are other GPU jobs running?")
                    return


                #print(sorted(list(history.history.keys())))
                plt.figure(figsize=(16,5))
                plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae'])

                plt.figure(figsize=(12,7))
                _P = model.keras_model.predict(X_val[:5])

                plot_some(X_val[:5], Y_val[:5], _P, pmax=99.5, cmap="gray")
                plt.suptitle('5 example validation patches\n'
                            'top row: input (source),  '
                            'middle row: target (ground truth),  '
                            'bottom row: predicted from source');

                plt.show()

                print("-- Export model for use in Fiji...")
                model.export_TF()
                print("Done")



    def predict_multiple(self, file_fns, n_tiles=(1,4,4), keep_meta=True):
        file_list = file_fns.split(";")
        for file_fn in file_list:
            self.predict(file_fn.strip(), n_tiles=n_tiles, keep_meta=keep_meta)

    def predict(self, file_fn, n_tiles=(1,4,4), keep_meta=True):
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

        series = 0
        z_size = reader.getSizeZ()
        y_size = reader.getSizeY()
        x_size = reader.getSizeX()
        c_size = reader.getSizeC()
        t_size = reader.getSizeT()

        z_out_size = int(z_size * self.low_scaling[0])
        y_out_size = int(y_size * self.low_scaling[1])
        x_out_size = int(x_size * self.low_scaling[2])

        if c_size != len(self.train_channels):
            print(" -- Warning: Number of Channels during training and prediction do not match. Using channels {} for prediction".format(self.train_channels))

        for ch in self.train_channels:
            model = CARE(None, 'CH_{}_model'.format(ch), basedir=pathlib.Path(self.out_dir) / 'models')
            res_image_ch = numpy.zeros(shape=(t_size, z_out_size, 1, y_out_size, x_out_size), dtype=dtype)
            print(" -- Predicting channel {}".format(ch))
            for t in tqdm(range(t_size), total=t_size):
                img_3d = numpy.zeros((z_size, y_size, x_size), dtype=dtype)
                for z in range(z_size):
                    img_3d[z, :, :] = ir.read(series=series,
                                                z=z,
                                                c=ch,
                                                t=t, rescale=False)

                img_3d_ch_ex = rescale(img_3d, self.low_scaling, preserve_range=True,
                                        order=self.order,
                                        multichannel=False,
                                        mode="reflect",
                                        anti_aliasing=True)

                pred = model.predict(img_3d_ch_ex, axes='ZYX', n_tiles=n_tiles)

                di = numpy.iinfo(dtype)
                pred = pred.clip(di.min, di.max).astype(dtype)

                res_image_ch[t, :, 0, :, :] = pred

                if False:
                    ch_t_out_fn = os.path.join(os.path.dirname(file_fn), os.path.splitext(os.path.basename(file_fn))[0] + "_care_predict_tp{:04d}_ch{}.tif".format(t, ch))
                    print("Saving time-point {} and channel {} to file '{}'".format(t, ch, ch_t_out_fn))
                    tifffile.imsave(ch_t_out_fn, pred[None,:, None, :, :], imagej=True, metadata={'axes': 'TZCYX'})



            ch_out_fn = os.path.join(os.path.dirname(file_fn),
                                     os.path.splitext(os.path.basename(file_fn))[0]
                                     + "_care_predict_ch{}.tif".format(ch))
            print(" -- Saving channel {} CARE prediction to file '{}'".format(ch, ch_out_fn))

            if keep_meta:
                reso      = (1 / (pixel_reso.X / self.low_scaling[2]),
                             1 / (pixel_reso.Y / self.low_scaling[1]))
                spacing   = pixel_reso.Z / self.low_scaling[0]
                unit      = pixel_reso.Xunit
                finterval = pixel_reso.T

                tifffile.imsave(ch_out_fn, res_image_ch, imagej=True, resolution=reso, metadata={'axes'     : 'TZCYX',
                                                                                                 'finterval': finterval,
                                                                                                 'spacing'  : spacing,
                                                                                                 'unit'     : unit})
            else:
                tifffile.imsave(ch_out_fn, res_image_ch)

            res_image_ch = None # should trigger gc and free the memory






def get_args():
    """
    Helper function for the argument parser.
    """

    description = """CAREless care: command line script for predicting new images given existing project."""

    parser = argparse.ArgumentParser(description=description)

    # Add arguments
    parser.add_argument('--care_project',  type=str, nargs=1, help="CAREless care project file (.json)", required=True)
    parser.add_argument('files',  type=str, nargs="+", help="Files to predict")
    parser.add_argument('--ntiles', nargs=3, type=int, default=[1, 8, 8])

    args = parser.parse_args()

    return args

def cmd_line():
    from .care import GuiParams

    args = get_args()
    assert os.path.exists(args.care_project[0]), f"Project file '{args.care_project[0]}'' does not exist."

    params = GuiParams()
    params.load(args.care_project[0])

    print("\n\nCAREless care parameters")
    print("-"*50)
    for k, v in params.items():
        print("{:25s}: {}".format(k,v))
    print("n-tiles:", args.ntiles)

    print("\n")

    for fn in args.files:
        assert os.path.exists(fn), f"File for prediction {fn} does not exist"

    print("Predicting Careless...")

    bt = CareTrainer(**params)
    for fn in args.files:
        bt.predict(fn, n_tiles=args.ntiles)

    JVM().shutdown()






