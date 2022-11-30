import os
import sys
import json
import glob
import time
import numpy

import pathlib
import tifffile
import ipywidgets as widgets
from functools import partial
from IPython.display import display

import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from ..care.dialogs import gui_dirname

from ..care.utils import (
    BFListReader,
    get_pixel_dimensions,
    get_space_time_resolution,
    get_file_list,
    JVM,
)
from ..care.care import (
    GuiParams,
    select_project,
    select_train_paramter,
    select_file_to_predict,
)

description = """Careless Noise2Void interface"""


class GuiParamsN2V(GuiParams):
    def initialize(self):
        self.clear()
        self["name"] = "careless_n2v"
        self["in_dir"] = "."
        self["glob"] = ""
        self["axes"] = "ZYX"
        self["patch_size"] = []
        self["n_patches_per_image"] = -1
        self["train_channels"] = [0]
        self["train_epochs"] = 40
        self["train_steps_per_epoch"] = 100
        self["train_batch_size"] = 16
        self["n2v_perc_pix"] = 1.5
        self["n2v_patch_shape"] = []
        self["n2v_neighborhood_radius"] = 5
        self["augment"] = False
        self["structuredN2V"] = "None"


params = GuiParamsN2V()
params.initialize()

select_project = partial(
    select_project, default_name="./careless_n2v.json", params=params
)
select_train_paramter = partial(select_train_paramter, params=params)


def select_input(params=params):
    if not params.is_loaded():
        print("Please create or load a project first..")
        return

    ### Input directory
    ###################
    btn_in_dir = widgets.Button(description="Select input folder")
    text_in_dir = widgets.Label(
        params["in_dir"], layout={"border": "1px solid black", "width": "400px"}
    )

    def btn_out_in_dir_clicked(btn):
        dir_name = gui_dirname(title="Choose input directory")
        text_in_dir.value = dir_name
        params["in_dir"] = dir_name

    btn_in_dir.on_click(btn_out_in_dir_clicked)
    select_in_dir = widgets.HBox([text_in_dir, btn_in_dir])

    select_directories = widgets.VBox([select_in_dir])

    ### Wildcard selection
    ######################
    def gui_select_files_widget(name, key):
        text_wc_low = widgets.Text(
            placeholder="*{}*".format(key),
            layout={"border": "1px solid black", "width": "100px"},
        )
        out_files_low = widgets.Output(
            layout={"border": "1px solid black", "width": "800px", "min_height": "40px"}
        )

        @out_files_low.capture(clear_output=True, wait=True)
        def text_wc_low_changed(change):
            if not change.new:
                return
            fl = get_file_list(text_in_dir.value, change.new)
            if len(fl) == 0:
                print("no files match...")
            else:
                for f in fl:
                    print(f.name)
                params[key] = change.new

        text_wc_low.observe(text_wc_low_changed, "value")

        label_in_dir = widgets.Label()
        mylink = widgets.dlink((text_in_dir, "value"), (label_in_dir, "value"))
        file_select_widget = widgets.VBox(
            [
                widgets.Label(name),
                widgets.HBox([label_in_dir, widgets.Label("/"), text_wc_low]),
                out_files_low,
            ]
        )

        text_wc_low.value = params[key]
        return file_select_widget

        ### Convert button

    ##################
    btn_convert = widgets.Button(description="Check")
    text_convert_repy = widgets.Label(layout={"width": "500px"})

    out_convert = widgets.Output()

    @out_convert.capture(clear_output=True, wait=True)
    def btn_convert_clicked(btn):
        text_convert_repy.value = "Checking..."

        datagen = BFListReader()
        datagen.from_glob(params["in_dir"], params["glob"])

        check_ok, msg = True, "OK"
        try:
            datagen.check_dims_equal()
        except AssertionError as ae:
            check_ok, msg = False, str(ae)

        if not check_ok:
            text_convert_repy.value = msg
        else:
            pix_dim = get_pixel_dimensions(
                get_file_list(params["in_dir"], params["glob"])[0]
            )
            z_dim = pix_dim.z

            if z_dim == 1:
                params["axes"] = "YX"
            else:

                params["axes"] = "ZYX"

            params["train_channels"] = list(range(pix_dim.c))

            text_convert_repy.value = "Using Noise2void in {}D mode with {} channels".format(
                len(params["axes"]), pix_dim.c
            )

    btn_convert.on_click(btn_convert_clicked)

    ### Combine
    ###########
    file_select_low = gui_select_files_widget("Images:", "glob")

    display(
        widgets.VBox(
            [
                select_directories,
                file_select_low,
                widgets.HBox([btn_convert, text_convert_repy]),
                out_convert,
            ]
        )
    )


### channel select
##################
def select_channel():
    available_channels = list(
        range(
            get_pixel_dimensions(get_file_list(params["in_dir"], params["glob"])[0]).c
        )
    )

    available_channels_str = list(map(str, available_channels))
    channel_str = list(map(str, params["train_channels"]))

    ms_channel = widgets.widgets.SelectMultiple(
        options=available_channels_str,
        value=channel_str,
        rows=len(available_channels_str),
    )

    def on_channel_change(change):
        params["train_channels"] = list(map(int, change.new))

    ms_channel.observe(on_channel_change, "value")
    ms_channel.value = channel_str
    display(
        widgets.HBox([widgets.Label("Channels", layout={"width": "200px"}), ms_channel])
    )


def select_patch_parameter():
    ### Path size select
    ####################
    patch_size_select = []
    patch_options = [8, 16, 32, 64, 128, 256]

    if len(params["patch_size"]) == 0:
        params["patch_size"] = [64] * len(list(params["axes"]))

    for j, a in enumerate(list(params["axes"])):
        wi = widgets.Dropdown(
            options=list(map(str, patch_options)),
            value=str(params["patch_size"][j]),
            desciption=a,
            layout={"width": "60px"},
        )

        def tmp_f(c, jj):
            params["patch_size"][jj] = int(c.new)

        wi.observe(partial(tmp_f, jj=j), "value")

        patch_size_select.append(widgets.Label(a))
        patch_size_select.append(wi)

    patch_size_select = widgets.HBox(patch_size_select)

    display(
        widgets.HBox(
            [widgets.Label("Patch size", layout={"width": "200px"}), patch_size_select]
        )
    )


def select_npatch_per_image():
    dd_n_patch_per_img = widgets.BoundedIntText(
        min=-1, max=4096 * 10, step=1, value=params["n_patches_per_image"]
    )

    def on_n_patch_per_img_change(change):
        params["n_patches_per_image"] = change.new

    dd_n_patch_per_img.observe(on_n_patch_per_img_change, "value")

    display(
        widgets.HBox(
            [
                widgets.Label(
                    "#Patches per image (-1: all)", layout={"width": "200px"}
                ),
                dd_n_patch_per_img,
            ]
        )
    )


def select_augment():
    dd_augment = widgets.Dropdown(
        options=[("8 rotation/flips", True), ("No augment", False)],
        value=params["augment"],
    )

    def on_dd_augment_change(change):
        params["augment"] = change.new

    dd_augment.observe(on_dd_augment_change, "value")

    display(
        widgets.HBox([widgets.Label("Augment", layout={"width": "200px"}), dd_augment])
    )


def select_n2v_parameter():
    ### N2V neighbor radius
    ###################
    int_n2v_neighborhood_radius = widgets.BoundedIntText(
        min=1, max=4096, step=1, value=params["n2v_neighborhood_radius"]
    )

    def on_n2v_neighborhood_radius_change(change):
        params["n2v_neighborhood_radius"] = change.new

    int_n2v_neighborhood_radius.observe(on_n2v_neighborhood_radius_change, "value")

    ### N2V perc pixel
    ###################
    float_n2v_perc_pix = widgets.BoundedFloatText(
        min=0, max=100, step=0.001, value=params["n2v_perc_pix"]
    )

    def on_n2v_perc_pix_change(change):
        params["n2v_perc_pix"] = change.new

    float_n2v_perc_pix.observe(on_n2v_perc_pix_change, "value")

    text_n2v_name = widgets.Text(value=params["name"])

    def on_text_n2v_name_change(change):
        params["name"] = change.new

    text_n2v_name.observe(on_text_n2v_name_change, "value")

    ### Struct N2V
    ##########################

    options = ["None", "horizontal", "vertical"]
    if len(params["axes"]) == 3:
        options.append("DeepInterpolation")

    dd_train_sn2v = widgets.Dropdown(options=options, value="None")

    def on_dd_train_sn2v_change(change):
        params["structuredN2V"] = change.new

    dd_train_sn2v.observe(on_dd_train_sn2v_change, "value")

    ### Combine
    ##############
    n2v_parameter = widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.Label("Neighborhood radius", layout={"width": "200px"}),
                    int_n2v_neighborhood_radius,
                ]
            ),
            widgets.HBox(
                [
                    widgets.Label(
                        "Percent pixel manipulation", layout={"width": "200px"}
                    ),
                    float_n2v_perc_pix,
                ]
            ),
            widgets.HBox(
                [widgets.Label("Model name", layout={"width": "200px"}), text_n2v_name]
            ),
            widgets.HBox(
                [
                    widgets.Label("Structured N2V", layout={"width": "200px"}),
                    dd_train_sn2v,
                ]
            ),
        ]
    )

    display(n2v_parameter)


def create_structN2Vmask(typ, patch_size):
    structN2Vmask = numpy.zeros(tuple(numpy.array(patch_size) - 1), dtype="uint8")
    if structN2Vmask.ndim == 2:
        structN2Vmask = structN2Vmask[None, ...]
    mask_shape = structN2Vmask.shape

    if typ == "None":
        return None
    elif typ == "DeepInterplation":
        structN2Vmask[mask_shape[0] // 2, :, :] = 1
    elif typ == "horizontal":
        structN2Vmask[
            :, mask_shape[1] // 2, mask_shape[2] // 4 : -mask_shape[2] // 4
        ] = 1
    elif typ == "vertical":
        structN2Vmask[
            :, mask_shape[1] // 4 : -mask_shape[1] // 4, mask_shape[2] // 2
        ] = 1

    return structN2Vmask.squeeze().tolist()


def train_predict(
    n_tiles=(1, 4, 4), params=params, files=None, headless=False, **unet_config
):
    """
    These advanced options can be set by keyword arguments:

    n_tiles : tuple(int)
        Number of tiles to tile the image into, if it is too large for memory.
    unet_residual : bool
        Parameter `residual` of :func:`n2v_old.nets.common_unet`. Default: ``n_channel_in == n_channel_out``
    unet_n_depth : int
        Parameter `n_depth` of :func:`n2v_old.nets.common_unet`. Default: ``2``
    unet_kern_size : int
        Parameter `kern_size` of :func:`n2v_old.nets.common_unet`. Default: ``5 if n_dim==2 else 3``
    unet_n_first : int
        Parameter `n_first` of :func:`n2v_old.nets.common_unet`. Default: ``32``
    batch_norm : bool
        Activate batch norm
    unet_last_activation : str
        Parameter `last_activation` of :func:`n2v_old.nets.common_unet`. Default: ``linear``
    train_learning_rate : float
        Learning rate for training. Default: ``0.0004``
    n2v_patch_shape : tuple
        Random patches of this shape are extracted from the given training data. Default: ``(64, 64) if n_dim==2 else (64, 64, 64)``
    n2v_manipulator : str
        Noise2Void pixel value manipulator. Default: ``uniform_withCP``
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable. Default: ``{'factor': 0.5, 'patience': 10}``
    n2v_manipulator : str
        Noise2Void pixel value manipulator. Default: ``uniform_withCP``
    """
    from n2v.models import N2VConfig, N2V

    from n2v.utils.n2v_utils import manipulate_val_data
    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

    if not headless:
        from csbdeep.utils import plot_history
        from matplotlib import pyplot as plt

    np = numpy

    # Init reader
    datagen = BFListReader()
    print("Loading images ...")
    if files is None:
        datagen.from_glob(params["in_dir"], params["glob"])
        files = datagen.get_file_names()
    else:
        datagen.from_file_list(files)

    print("Training ...")
    for c in params["train_channels"]:
        print("  -- Channel {}".format(c))

        imgs_for_patches = datagen.load_imgs_generator()
        imgs_for_predict = datagen.load_imgs_generator()

        img_ch = (im[..., c : c + 1] for im in imgs_for_patches)
        img_ch_predict = (im[..., c : c + 1] for im in imgs_for_predict)

        npatches = (
            params["n_patches_per_image"] if params["n_patches_per_image"] > 1 else None
        )

        patches = N2V_DataGenerator().generate_patches_from_list(
            img_ch,
            num_patches_per_img=npatches,
            shape=params["patch_size"],
            augment=params["augment"],
        )

        numpy.random.shuffle(patches)

        sep = int(len(patches) * 0.9)
        X = patches[:sep]
        X_val = patches[sep:]

        # Structured N2V
        if "structN2Vmask" in unet_config.keys():
            structN2Vmask = unet_config.pop("structN2Vmask")
        else:
            structN2Vmask = create_structN2Vmask(
                params["structuredN2V"], params["patch_size"]
            )

        config = N2VConfig(
            X,
            train_steps_per_epoch=params["train_steps_per_epoch"],
            train_epochs=params["train_epochs"],
            train_loss="mse",
            train_batch_size=params["train_batch_size"],
            n2v_perc_pix=params["n2v_perc_pix"],
            n2v_patch_shape=params["patch_size"],
            n2v_manipulator="uniform_withCP",
            n2v_neighborhood_radius=params["n2v_neighborhood_radius"],
            structN2Vmask=structN2Vmask,
            **unet_config,
        )

        # a name used to identify the model
        model_name = "{}_ch{}".format(params["name"], c)
        # the base directory in which our model will live
        basedir = "models"
        # We are now creating our network model.
        model = N2V(config=config, name=model_name, basedir=params["in_dir"])

        history = model.train(X, X_val)

        val_patch = X_val[0, ..., 0]
        val_patch_pred = model.predict(val_patch, axes=params["axes"])

        if "Z" in params["axes"]:
            val_patch = val_patch.max(0)
            val_patch_pred = val_patch_pred.max(0)

        if not headless:
            f, ax = plt.subplots(1, 2, figsize=(14, 7))
            ax[0].imshow(val_patch, cmap="gray")
            ax[0].set_title("Validation Patch")
            ax[1].imshow(val_patch_pred, cmap="gray")
            ax[1].set_title("Validation Patch N2V")

            plt.figure(figsize=(16, 5))
            plot_history(history, ["loss", "val_loss"])

        print("  -- Predicting channel {}".format(c))
        for f, im in zip(files, img_ch_predict):
            print("  -- {}".format(f))
            pixel_reso = get_space_time_resolution(str(f))
            res_img = []
            for t in range(len(im)):
                nt = n_tiles if "Z" in params["axes"] else n_tiles[1:]
                pred = model.predict(im[t, ..., 0], axes=params["axes"], n_tiles=nt)

                if "Z" in params["axes"]:
                    pred = pred[:, None, ...]
                res_img.append(pred)

            pred = numpy.stack(res_img)
            if "Z" not in params["axes"]:
                pred = pred[:, None, None, ...]

            reso = (1 / pixel_reso.X, 1 / pixel_reso.Y)
            spacing = pixel_reso.Z
            unit = pixel_reso.Xunit
            finterval = pixel_reso.T

            tifffile.imsave(
                "{}_n2v_pred_ch{}.tiff".format(str(f)[:-4], c),
                pred,
                imagej=True,
                resolution=reso,
                metadata={
                    "axes": "TZCYX",
                    "finterval": finterval,
                    "spacing": spacing,
                    "unit": unit,
                },
            )


def predict(files, n_tiles=(1, 4, 4), params=params):
    from n2v.models import N2V

    files = [f.strip() for f in files.split(";")]
    datagen = BFListReader()
    datagen.from_file_list(files)

    axes = datagen.check_dims_equal()
    axes = axes.replace("T", "").replace("C", "")

    assert (
        axes == params["axes"]
    ), "The files to predict have different dimensionality: {} != {}".format(
        axes, params["axes"]
    )

    print("Predicting ...")
    for c in params["train_channels"]:
        imgs = datagen.load_imgs_generator()
        print("  -- Channel {}".format(c))

        img_ch = (im[..., c : c + 1] for im in imgs)
        # a name used to identify the model
        model_name = "{}_ch{}".format(params["name"], c)
        # We are now creating our network model.
        model = N2V(config=None, name=model_name, basedir=params["in_dir"])

        for f, im in zip(files, img_ch):
            print("  -- {}".format(f))
            pixel_reso = get_space_time_resolution(str(f))
            res_img = []
            for t in range(len(im)):
                nt = n_tiles if "Z" in params["axes"] else n_tiles[1:]
                pred = model.predict(im[t, ..., 0], axes=params["axes"], n_tiles=nt)

                if "Z" in params["axes"]:
                    pred = pred[:, None, ...]
                res_img.append(pred)

            pred = numpy.stack(res_img)
            if "Z" not in params["axes"]:
                pred = pred[:, None, None, ...]

            reso = (1 / pixel_reso.X, 1 / pixel_reso.Y)
            spacing = pixel_reso.Z
            unit = pixel_reso.Xunit
            finterval = pixel_reso.T

            tifffile.imsave(
                "{}_n2v_pred_ch{}.tiff".format(str(f)[:-4], c),
                pred,
                imagej=True,
                resolution=reso,
                metadata={
                    "axes": "TZCYX",
                    "finterval": finterval,
                    "spacing": spacing,
                    "unit": unit,
                },
            )


def cmd_line():
    import argparse

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
        "files", type=str, nargs="*", help="Files to process with Noise2Void"
    )
    parser.add_argument("--ntiles", nargs=3, type=int, default=[1, 8, 8])
    parser.add_argument("--model_name", nargs=1, type=str, default=None)

    args = parser.parse_args()
    params.load(args.n2v_project[0])
    print("\n\CAREless Noise2Void parameters")
    print("-" * 50)
    for k, v in params.items():
        print("{:25s}: {}".format(k, v))
    print("n-tiles:", args.ntiles)

    print("\n")

    files = args.files
    if len(files) == 0:
        files = None

    model_name = args.model_name
    if model_name is not None:
        params["name"] = model_name[0]

    train_predict(params=params, files=files, n_tiles=tuple(args.ntiles), headless=True)

    JVM().shutdown()
    sys.exit(0)

