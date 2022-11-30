import os
import json
import numpy
import pathlib
import ipywidgets as widgets
from functools import partial
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from csbdeep.utils import plot_some

from .dialogs import gui_fname
from .dialogs import gui_dirname
from .dialogs import gui_fsavename
from .dialogs import gui_fnames
from .core import CareInputConverter, CareTrainer
from .utils import get_pixel_dimensions, get_file_list, get_upscale_factors, \
                   check_file_lists, get_space_time_resolution

from matplotlib import pyplot as plt

# global parameters
class GuiParams(dict):
    _file = None
    def load(self, proj_fn):
        with open(proj_fn, "r") as read_file:
            params = json.load(read_file)
            for k, v in params.items():
                self[k] = v
        self.__class__._file = proj_fn

    def saveas(self, proj_fn):
        self.__class__._file = proj_fn
        self.save()

    def save(self):
        with open(self.__class__._file, "w") as save_file:
            json.dump(self, save_file)

    def is_loaded(self):
        return self.__class__._file is not None

    def initialize(self):
        self.clear()
        self["in_dir"]  = "."
        self["out_dir"] = "."
        self["low_wc"] = ""
        self["high_wc"] = ""
        self["axes"]   = "ZYX"
        self['patch_size'] = []
        self['n_patches_per_image'] = 128
        self["train_channels"] = [0]
        self['train_epochs'] = 40
        self['train_steps_per_epoch'] = 100
        self['train_batch_size'] = 16
        self['probabilistic'] = False



params = GuiParams()
params.initialize()

### GUI widgets
def select_project(default_name="./careless_care.json", params=params):
    btn_new_proj = widgets.Button(description="New")
    btn_load_proj = widgets.Button(description="Load")

    out_project = widgets.Output()

    @out_project.capture(clear_output=True, wait=True)
    def btn_btn_new_proj_clicked(btn):
        new_proj_fn = gui_fsavename(title="Create CAREless project", initialfile=default_name,  filetypes=[("JSON", "*.json")])
        if len(new_proj_fn) == 0:
            return
        params.initialize()
        params.saveas(new_proj_fn)
        print("New project initialized: '{}'".format(new_proj_fn))
        params["name"] = os.path.splitext( os.path.basename(new_proj_fn))[0]
    btn_new_proj.on_click(btn_btn_new_proj_clicked)

    @out_project.capture(clear_output=True, wait=True)
    def btn_btn_load_proj_clicked(btn):
        proj_fn = gui_fname(title="Load CAREless project", filetypes=[("JSON", "*.json")])
        if len(proj_fn) == 0:
            return
        params.load(proj_fn)
        print("Project loaded: '{}'".format(proj_fn))
    btn_load_proj.on_click(btn_btn_load_proj_clicked)

    display(widgets.VBox([ widgets.HBox([widgets.Label("Project:"), btn_new_proj, btn_load_proj]),
                            out_project]))


def select_input():
    if not params.is_loaded():
         print("Please create or load a project first..")
         return

    ### Input directory
    ###################
    btn_in_dir = widgets.Button(description="Select input folder")
    text_in_dir = widgets.Label(params["in_dir"], layout={'border': '1px solid black', "width":"400px"})

    def btn_out_in_dir_clicked(btn):
        dir_name = gui_dirname(title="Choose input directory")
        text_in_dir.value = dir_name
        params["in_dir"] = dir_name

    btn_in_dir.on_click(btn_out_in_dir_clicked)
    select_in_dir = widgets.HBox([text_in_dir, btn_in_dir])

    ### Output directory
    #####################
    btn_out_dir = widgets.Button(description="Select output folder")
    text_out_dir = widgets.Label(params["out_dir"], layout={'border': '1px solid black', "width":"400px"})

    def btn_out_out_dir_clicked(btn):
        dir_name = gui_dirname(title="Choose input directory")
        text_out_dir.value = dir_name
        params["out_dir"] = dir_name

    btn_out_dir.on_click(btn_out_out_dir_clicked)
    select_out_dir = widgets.HBox([text_out_dir, btn_out_dir])

    select_directories = widgets.VBox([select_in_dir, select_out_dir])

    ### Wildcard selection
    ######################
    def gui_select_files_widget(name, key):
        text_wc_low = widgets.Text(placeholder="*{}*".format(key), layout={'border': '1px solid black', "width":"100px"})
        out_files_low = widgets.Output(layout={'border': '1px solid black', "width":"500px", "min_height": "40px"})

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


        text_wc_low.observe(text_wc_low_changed, 'value')


        label_in_dir = widgets.Label()
        mylink = widgets.dlink((text_in_dir, 'value'), (label_in_dir, 'value'))
        file_select_widget = widgets.VBox([widgets.Label(name),
                                    widgets.HBox([label_in_dir,
                                                widgets.Label("/"), text_wc_low]),
                                    out_files_low])

        text_wc_low.value = params[key]
        return file_select_widget

    ### Convert button
    ##################
    btn_convert = widgets.Button(description="Check & Convert")
    text_convert_repy = widgets.Label(layout={"width":"500px"})

    out_convert = widgets.Output()

    @out_convert.capture(clear_output=True, wait=True)
    def btn_convert_clicked(btn):
        text_convert_repy.value = "Checking..."
        check_ok, msg = check_file_lists(params["in_dir"], params["low_wc"], params["high_wc"])
        if not check_ok:
            text_convert_repy.value = msg
        else:
            params["low_scaling"] = get_upscale_factors(params["in_dir"], params["low_wc"], params["high_wc"])

            z_dim = get_pixel_dimensions(get_file_list(params["in_dir"], params["low_wc"])[0]).z

            if z_dim == 1:
                params["axes"] = "YX"
            else:

                params["axes"] = "ZYX"

            print("Using CARE in {}D mode".format(len(params["axes"])))

            if (numpy.array(params["low_scaling"]) == 1).all():
                text_convert_repy.value = "Low quality images match high quality resolution"
            else:
                text_convert_repy.value = "Low quality images are scaled up by ({}, {}, {}) in ZYX to match high quality resolution".format(*params["low_scaling"])

            CareInputConverter(in_dir =params["in_dir"],
                                  out_dir=params["out_dir"],
                                  low_wc =params["low_wc"] ,
                                  high_wc=params["high_wc"]).convert()

    btn_convert.on_click(btn_convert_clicked)

    ### Combine
    ###########
    file_select_low = gui_select_files_widget("Images: low quality", "low_wc")
    file_select_high = gui_select_files_widget("Images: high quality", "high_wc")
    select_files = widgets.HBox([file_select_low, file_select_high])

    display(widgets.VBox([select_directories, select_files, widgets.HBox([btn_convert, text_convert_repy]), out_convert]))





### channel select
##################
def gui_select_channel_widget():
    available_channels = list(range(get_pixel_dimensions(get_file_list(params["in_dir"], params["low_wc"])[0]).c))
    available_channels_str = list(map(str, available_channels))

    channel_str = list(map(str, params["train_channels"]))

    ms_channel = widgets.widgets.SelectMultiple(
                                            options=available_channels_str,
                                            value=channel_str,
                                            rows=len(available_channels_str),
                                        )
    def on_channel_change(change):
        params['train_channels'] = list(map(int, change.new))

    ms_channel.observe(on_channel_change, 'value')
    ms_channel.value = channel_str
    return ms_channel


def select_patch_parameter():
    ### Path size select
    ####################
    patch_size_select = []
    patch_options = [8, 16, 32, 64, 128, 256]

    if len(params['patch_size']) == 0:
        params['patch_size'] = [64]*len(list(params["axes"]))

    for j, a in enumerate(list(params["axes"])):
        wi = widgets.Dropdown(options=list(map(str, patch_options)),
                            value=str(params['patch_size'][j]),
                            desciption=a,
                            layout={'width':'60px'})

        def tmp_f(c, jj):
            params['patch_size'][jj]=int(c.new)
        wi.observe(partial(tmp_f, jj=j), 'value')


        patch_size_select.append(widgets.Label(a))
        patch_size_select.append(wi)

    patch_size_select = widgets.HBox(patch_size_select)


    ### Number of patches per image
    ###############################

    dd_n_patch_per_img = widgets.BoundedIntText(min=1, max=4096,step=1,
                                          value=params['n_patches_per_image'])

    def on_n_patch_per_img_change(change):
        params['n_patches_per_image'] = change.new

    dd_n_patch_per_img.observe(on_n_patch_per_img_change, 'value')

    ms_channel = gui_select_channel_widget()


    patch_parameter = widgets.VBox(
                               [widgets.HBox([widgets.Label('Channels', layout={'width':'200px'}), ms_channel]),
                                widgets.HBox([widgets.Label('Patch size', layout={'width':'200px'}), patch_size_select]),
                                widgets.HBox([widgets.Label('#Patches / image', layout={'width':'200px'}), dd_n_patch_per_img]),
                                ])

    display(patch_parameter)

### Probabilistic
###################
def select_probabilistic(params=params):
    dd_train_proba = widgets.Dropdown(options=[False, True], value=params['probabilistic'])

    def on_dd_train_proba_change(change):
        params['probabilistic'] = change.new

    dd_train_proba.observe(on_dd_train_proba_change, 'value')

    probab = widgets.VBox([widgets.HBox([widgets.Label('Probabilistic', layout={'width':'200px'}), dd_train_proba]),])

    display(widgets.HTML(value="<i><b>experimental</b>, prediction output will contain mean and scale as channels</i>"))
    display(probab)


### Train parameter
###################
def select_train_paramter(params=params):
    ### Training epochs
    ###################
    int_train_epochs = widgets.BoundedIntText(min=1, max=4096, step=1, value=params['train_epochs'])

    def on_int_train_epochs_change(change):
        params['train_epochs'] = change.new

    int_train_epochs.observe(on_int_train_epochs_change, 'value')

    ### Steps per epoch
    ###################
    int_train_steps_per_epoch = widgets.BoundedIntText(min=1, max=4096, step=1, value=params['train_steps_per_epoch'])

    def on_train_steps_per_epoch_change(change):
        params['train_steps_per_epoch'] = change.new

    int_train_steps_per_epoch.observe(on_train_steps_per_epoch_change, 'value')


    ### Batch size
    ##############
    dd_train_batch_size = widgets.Dropdown(options=[4, 8, 16, 32, 64, 128, 256], value=params['train_batch_size'])

    def on_dd_train_batch_size_change(change):
        params['train_batch_size'] = change.new

    dd_train_batch_size.observe(on_dd_train_batch_size_change, 'value')



    ### Combine
    ##############
    train_parameter = widgets.VBox(
                               [
                                widgets.HBox([widgets.Label('#Epochs', layout={'width':'200px'}), int_train_epochs]),
                                widgets.HBox([widgets.Label('#Steps / epoch', layout={'width':'200px'}), int_train_steps_per_epoch]),
                                widgets.HBox([widgets.Label('Batch size', layout={'width':'200px'}), dd_train_batch_size]),

                                ])

    display(train_parameter)


def select_file_to_predict():
    btn_predict_file = widgets.Button(description="Select file(s) for prediction")
    text_predict_fn  = widgets.Textarea("", layout={'border': '1px solid black', "width":"800px", 'height': '100%'})
    out_predict_fn   = widgets.Output(layout={ "width":"800px", "min_height": "40px"})

    @out_predict_fn.capture(clear_output=True, wait=True)
    def btn_predict_file_clicked(btn):
        predict_fn = gui_fnames(title="Select images to predcit")
        #print(predict_fn)
        text_predict_fn.value = "\n".join(predict_fn)


    btn_predict_file.on_click(btn_predict_file_clicked)
    predict_file_widget = widgets.HBox([text_predict_fn, btn_predict_file])
    display(widgets.VBox([predict_file_widget, out_predict_fn]))
    return text_predict_fn





