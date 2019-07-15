import os
import json
import glob
import numpy
import pathlib
import tifffile
import ipywidgets as widgets
from functools import partial
from IPython.display import display


from bif_care.qt_file_dialog import gui_fname
from bif_care.qt_dir_dialog import gui_dirname
from bif_care.qt_files_dialog import gui_fnames
from bif_care.qt_filesave_dialog import gui_fsavename
from bif_care.gui import GuiParams, select_project, select_train_paramter
from bif_care.utils import get_pixel_dimensions, get_space_time_resolution, get_file_list

class GuiParamsN2V(GuiParams):
    def initialize(self):
        self.clear()
        self['name'] = "bif_n2v"
        self["in_dir"]  = "."
        self["glob"] = ""
        self["axes"]   = "ZYX"
        self['patch_size'] = []
        self['n_patches_per_image'] = -1
        self["train_channels"] = [0]
        self['train_epochs'] = 40
        self['train_steps_per_epoch'] = 100
        self['train_batch_size'] = 16
        self['n2v_perc_pix'] = 0.016
        self['n2v_patch_shape'] = []
        self['n2v_neighborhood_radius'] = 5
        

params = GuiParamsN2V()
params.initialize()

params.load("C:/Users/csommer/Desktop/bif_n2v2.json")

select_project = partial(select_project, default_name='./bif_n2v.json', params=params)
select_train_paramter = partial(select_train_paramter, params=params)

def select_input(params=params):
    ### Input directory
    ###################
    btn_in_dir = widgets.Button(description="Select input folder")
    text_in_dir = widgets.Label(params["in_dir"], layout={'border': '1px solid black', "width":"400px"})

    def btn_out_in_dir_clicked(btn):
        dir_name = gui_dirname()
        text_in_dir.value = dir_name
        params["in_dir"] = dir_name

    btn_in_dir.on_click(btn_out_in_dir_clicked)       
    select_in_dir = widgets.HBox([text_in_dir, btn_in_dir])   

    select_directories = widgets.VBox([select_in_dir])

    ### Wildcard selection
    ######################
    def gui_select_files_widget(name, key):
        text_wc_low = widgets.Text(placeholder="*{}*".format(key), layout={'border': '1px solid black', "width":"100px"})
        out_files_low = widgets.Output(layout={'border': '1px solid black', "width":"800px", "min_height": "40px"})

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
    btn_convert = widgets.Button(description="Check")
    text_convert_repy = widgets.Label(layout={"width":"500px"})

    out_convert = widgets.Output()

    @out_convert.capture(clear_output=True, wait=True)
    def btn_convert_clicked(btn):
        text_convert_repy.value = "Checking..."
        ###TODO Check
        check_ok, msg = True, "OK"
        if not check_ok: 
            text_convert_repy.value = msg
        else:

            z_dim = get_pixel_dimensions(get_file_list(params["in_dir"], params["glob"])[0]).z

            if z_dim == 1:
                params["axes"] = "YX"
            else:
                
                params["axes"] = "ZYX"

            text_convert_repy.value = "Using CARE in {}D mode".format(len(params["axes"]))

    btn_convert.on_click(btn_convert_clicked)     

    ### Combine
    ###########
    file_select_low = gui_select_files_widget("Images:", "glob")


    display(widgets.VBox([select_directories, file_select_low, widgets.HBox([btn_convert, text_convert_repy]), out_convert]))


### channel select
################## 
def select_channel():
    available_channels = list(range(get_pixel_dimensions(get_file_list(params["in_dir"], params["glob"])[0]).c))
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
    display(widgets.HBox([widgets.Label("Channels", layout={'width':'100px'}), ms_channel]))

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

    display(widgets.HBox([widgets.Label('Patch size', layout={'width':'100px'}), patch_size_select]))


def select_npatch_per_image():
    dd_n_patch_per_img = widgets.BoundedIntText(min=-1, max=4096*2,step=1,
                                          value=params['n_patches_per_image'])

    def on_n_patch_per_img_change(change):
        params['n_patches_per_image'] = change.new

    dd_n_patch_per_img.observe(on_n_patch_per_img_change, 'value') 

    display(widgets.HBox([widgets.Label('#Patches per image', layout={'width':'100px'}), dd_n_patch_per_img]))

def select_n2v_parameter():
    ### N2V neighbor radius
    ###################
    int_n2v_neighborhood_radius = widgets.BoundedIntText(min=1, max=4096, step=1, value=params['n2v_neighborhood_radius'])
    
    def on_n2v_neighborhood_radius_change(change):
        params['n2v_neighborhood_radius'] = change.new

    int_n2v_neighborhood_radius.observe(on_n2v_neighborhood_radius_change, 'value')

    ### N2V perc pixel
    ###################
    float_n2v_perc_pix = widgets.BoundedFloatText(min=0, max=0.016, step=0.001, value=params['n2v_perc_pix'])
    
    def on_n2v_perc_pix_change(change):
        params['n2v_perc_pix'] = change.new

    float_n2v_perc_pix.observe(on_n2v_perc_pix_change, 'value')

    text_n2v_name = widgets.Text(value=params['name'])
    
    def on_text_n2v_name_change(change):
        params['name'] = change.new

    text_n2v_name.observe(on_text_n2v_name_change, 'value')



    ### Combine
    ##############
    n2v_parameter = widgets.VBox([
                                  widgets.HBox([widgets.Label('Neighborhood radius', layout={'width':'100px'}), int_n2v_neighborhood_radius]),
                                  widgets.HBox([widgets.Label('Perc. pixel manipulation', layout={'width':'100px'}), float_n2v_perc_pix]),
                                  widgets.HBox([widgets.Label('Model name', layout={'width':'100px'}), text_n2v_name]),
                                ])

                    

    display(n2v_parameter)

def train_predict(n_tiles=(1,4,4), params=params):
    from n2v.models import N2VConfig, N2V
    from csbdeep.utils import plot_history
    from n2v.utils.n2v_utils import manipulate_val_data
    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
    from matplotlib import pyplot as plt
    from bif_care.utils import BFListReader
    np = numpy

    datagen = BFListReader(params["in_dir"], params["glob"])
    files = datagen.img_fns

    imgs = datagen.load_imgs()

    for c in params["train_channels"]:
        print("Training channel {}".format(c))

        img_ch = [im[..., c:c+1] for im in imgs]

        npatches = params["n_patches_per_image"] if params["n_patches_per_image"] > 1 else None
        print("npatches", npatches)
        patches = N2V_DataGenerator().generate_patches_from_list(img_ch, num_patches_per_img=npatches, shape=params['patch_size'])

        sep = int(len(patches)*0.9)
        X     = patches[:sep]
        X_val = patches[ sep:]

        print("X.shape", X.shape)
        print(img_ch[0].shape, img_ch[0].shape)


        config = N2VConfig(X, 
                    unet_kern_size=3, 
                    train_steps_per_epoch=params["train_steps_per_epoch"],
                    train_epochs=params["train_epochs"], 
                    train_loss='mse', 
                    batch_norm=True, 
                    train_batch_size=params["train_batch_size"], 
                    n2v_perc_pix=params["n2v_perc_pix"], 
                    n2v_patch_shape=params['patch_size'], 
                    n2v_manipulator='uniform_withCP', 
                    n2v_neighborhood_radius=params["n2v_neighborhood_radius"])


        # a name used to identify the model
        model_name = '{}_ch{}'.format(params['name'], c)
        # the base directory in which our model will live
        basedir = 'models'
        # We are now creating our network model.
        model = N2V(config=config, name=model_name, basedir=params["in_dir"])

        history = model.train(X, X_val)

        val_patch = X_val[0,..., 0]
        print("val_patch.shape", val_patch.shape)

        val_patch_pred = model.predict(val_patch,axes=params["axes"])

        print("val_patch_pred.shape", val_patch_pred.shape)

        # Let's look at two patches.
        f, ax = plt.subplots(1,2, figsize=(14,7))

        ax[0].imshow(val_patch[0, ...],cmap='gray')
        ax[0].set_title('Validation Patch')

        ax[1].imshow(val_patch_pred[0, ...],cmap='gray')
        ax[1].set_title('Validation Patch N2V')


        plt.figure(figsize=(16,5))
        plot_history(history,['loss','val_loss'])

        for f, im in zip(files, img_ch):
            print("Predicting {}".format(f))
            pixel_reso = get_space_time_resolution(str(f))
            res_img = []
            for t in range(len(im)):
                pred = model.predict(im[t,..., 0], axes=params["axes"], n_tiles=n_tiles)
                pred = pred[:, None, ...]
                res_img.append(pred)

            pred = numpy.stack(res_img)

            reso      = (1 / pixel_reso.X, 
                         1 / pixel_reso.Y )
            spacing   = pixel_reso.Z
            unit      = pixel_reso.Xunit
            finterval = pixel_reso.T

            tifffile.imsave("{}_n2v_pred_ch{}.tiff".format(str(f)[:-4], c), pred, imagej=True, resolution=reso, metadata={'axes': 'TZCYX',
                                                                                                'finterval': finterval,
                                                                                                'spacing'  : spacing, 
                                                                                                'unit'     : unit})
