import os
import pathlib
import ipywidgets as widgets
from functools import partial
from IPython.display import display
from .qt_file_dialog import gui_fname
from  .qt_dir_dialog import gui_dirname
from .core import BifCareInputConverter
from .utils import get_pixel_dimensions, get_file_list, get_upscale_factors, check_file_lists

# global parameters
params = {"in_dir": "."}

def select_inputs_widget():
    ### Input directory
    ###################
    btn_in_dir = widgets.Button(description="Select input folder")
    text_in_dir = widgets.Label(".", layout={'border': '1px solid black', "width":"400px"})

    def btn_out_in_dir_clicked(btn):
        dir_name = gui_dirname()
        text_in_dir.value = dir_name
        params["in_dir"] = dir_name

    btn_in_dir.on_click(btn_out_in_dir_clicked)       
    select_in_dir = widgets.HBox([text_in_dir, btn_in_dir])   

    ### Output directory
    #####################
    btn_out_dir = widgets.Button(description="Select output folder")
    text_out_dir = widgets.Label(".", layout={'border': '1px solid black', "width":"400px"})

    def btn_out_out_dir_clicked(btn):
        dir_name = gui_dirname()
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
        text_wc_low.value = ""
        return file_select_widget

    ### Convert button
    ##################

    btn_convert = widgets.Button(description="Check & Convert")
    text_convert_repy = widgets.Label(layout={"width":"500px"})

    out_convert = widgets.Output()

    @out_convert.capture(clear_output=True, wait=True)
    def btn_convert_clicked(btn):
        check_ok, msg = check_file_lists(params["in_dir"], params["low_wc"], params["high_wc"])
        if not check_ok: 
            text_convert_repy.value = msg
        else:
            (z,y,x) = get_upscale_factors(params["in_dir"], params["low_wc"], params["high_wc"])
            params["low_scaling"] = [z,y,x]
            text_convert_repy.value = "Low quality images are scaled up by ({}, {}, {}) in ZYX to match high quality resolution".format(z,y,x)
            
            BifCareInputConverter(in_dir=params["in_dir"],
                                  out_dir=params["out_dir"],
                                  low_wc=params["low_wc"] ,
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
    channels = list(range(get_pixel_dimensions(get_file_list(params["in_dir"], params["low_wc"])[0]).c))
    channels_str = list(map(str, channels))
    ms_channel = widgets.widgets.SelectMultiple(
                                            options=channels_str,
                                            value=channels_str,
                                            rows=len(channels_str),  
                                        )
    def on_channel_change(change):
        params['train_channels'] = list(map(int, change.new))

    ms_channel.observe(on_channel_change, 'value')
    ms_channel.value = channels_str
    params["train_channels"] = channels
    return ms_channel

### Train parameter 
###################
def select_train_paramter_widget():
    ### Path size select
    ####################
    patch_size_select = []
    patch_options = [8, 16, 32, 64, 128]
    params['patch_size'] = [16, 64, 64] 
    for j, a in enumerate(['Z', 'Y', 'X']):
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

    dd_n_patch_per_img = widgets.Dropdown(options=[128, 256, 512, 1024, 2048],
                                        value=128)

    ### Number of patches per image
    ###############################
    params['n_patches_per_image'] = 128
    def on_n_patch_per_img_change(change):
        params['n_patches_per_image'] = change.new

    dd_n_patch_per_img.observe(on_n_patch_per_img_change, 'value')
    dd_n_patch_per_img.value = 128

    ### Training epochs
    ###################
    int_train_epochs = widgets.IntSlider(value=40, min=10, max=100, step=5,)
    params['train_epochs'] = 40
    def on_int_train_epochs_change(change):
        params['train_epochs'] = change.new

    int_train_epochs.observe(on_int_train_epochs_change, 'value')
    int_train_epochs.value = 40

    ### Steps per epoch
    ###################
    int_train_steps_per_epoch = widgets.IntSlider(value=100, min=10, max=400, step=10,)
    params['train_steps_per_epoch'] = 100
    def on_train_steps_per_epoch_change(change):
        params['train_steps_per_epoch'] = change.new

    int_train_steps_per_epoch.observe(on_train_steps_per_epoch_change, 'value')
    int_train_steps_per_epoch.value = 100
    
    
    ### Batch size
    ##############
    dd_train_batch_size = widgets.Dropdown(options=[8, 16, 32, 64, 128], value=16) 
    params['train_batch_size'] = 16
    def on_dd_train_batch_size_change(change):
        params['train_batch_size'] = change.new

    dd_train_batch_size.observe(on_dd_train_batch_size_change, 'value')
    dd_train_batch_size.value = 16

    ms_channel = gui_select_channel_widget()

    ### Combine
    ##############
    train_parameter = widgets.VBox(
                               [widgets.HBox([widgets.Label('Channels', layout={'width':'100px'}), ms_channel]), 
                                widgets.HBox([widgets.Label('Patch size', layout={'width':'100px'}), patch_size_select]), 
                                widgets.HBox([widgets.Label('#Patches / image', layout={'width':'100px'}), dd_n_patch_per_img]),

                                widgets.HBox([widgets.Label('#Epochs', layout={'width':'100px'}), int_train_epochs]),
                                widgets.HBox([widgets.Label('#Steps / epoch', layout={'width':'100px'}), int_train_steps_per_epoch]),
                                widgets.HBox([widgets.Label('Batch size', layout={'width':'100px'}), dd_train_batch_size]),
                                
                                ])

    display(train_parameter)

def select_project():
    btn_project = widgets.Button(description="Select BifCare project (bif_care.json)")
    if "out_dir" in params.keys() and os.path.exists(os.path.join(params["out_dir"], "bif_care.json")):
        project_fn = os.path.join(params["out_dir"], "bif_care.json")
    else:
        project_fn = ""

    text_project_fn = widgets.Label(project_fn, layout={'border': '1px solid black', "width":"400px"})

    def btn_project_clicked(btn):
        project_fn = gui_fname()
        text_project_fn.value = project_fn

    btn_project.on_click(btn_project_clicked)       
    display(widgets.HBox([text_project_fn, btn_project]))
    return text_project_fn

def select_file_to_predict(): 
    btn_predict_file = widgets.Button(description="Select file for prediction")
    text_predict_fn  = widgets.Label("", layout={'border': '1px solid black', "width":"400px"})
    
    def btn_predict_file_clicked(btn):
        predict_fn = gui_fname()
        text_predict_fn.value = predict_fn
    
    btn_predict_file.on_click(btn_predict_file_clicked)       
    predict_file_widget = widgets.HBox([text_predict_fn, btn_predict_file])
    display(predict_file_widget)
    return text_predict_fn

    



