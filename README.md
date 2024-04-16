# CARE-*less* care|n2v
---
Simple IPython based user interface to [CARE](http://csbdeep.bioimagecomputing.com/) a toolbox for Content-aware Image Restoration and to [Noise2Void](https://github.com/juglab/n2v) a self-supervised deep learning based denoising.

# Installation (care and n2v)
---
We strongly recommend using the [Anaconda Python distribution (64bit)](https://www.anaconda.com/distribution/) with Python == 3.9. Furthermore, we recommend to create a new conda virtual environment.

You can install CAREless and dependencies into a new conda environment by:

Download the `environment.yml` from this repository, then:

```
conda env create -f environment.yml
conda activate careless
```

## Windows 
You will need to install the *Microsoft Visual Studio C++ Build Tools* and the *Windows 10 SDK* prior to the conda installation. In the Visual Studio (Community) installer, select [individual components](https://seafile.ist.ac.at/f/029e5e1ca96b41f19e44/) and check:

* MSVC v143 - VS 2022 C++ x64/x86 build tool...
* Windows 10 SDK (10.0.20348.0)

> :warning:  Both are required to build the bioformats extension


## Linux 

You might have to set:

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/`

Verify the installation by:

`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

This should output your GPU device.

# CARE and N2V in Jupyter

### Download notebooks
Download the care and n2v notebook from the notebooks folder of this repository. Then you can start jupyterlab by:

```
# conda activate careless
jupyter lab <notebooks-folder>
```

This should open your browser with the jupyterlab IDE. Now you can select the care or the n2v notebook.

CARE needs pairs of registered images - low (input) and high (output) quality. It trains a convolutional neural network how to transform low quality images - which might even be of less physical resolution - into high quality images. After training, newly recorded low quality images or movies can be predicted. 2D, 3D and multi-channel images are supported. For each channel a separate network is trained.

### Vanilla screencast for input selection and training
[CAREless user interface](vid/bif_care_demo_01.mp4)

### Input selection and patch extraction
In order to train CARE, the path to the image pairs needs to be specified. Then, choose images for low and high quality respectively using a wild-card (e.g. `low*.tif` and `high*.tif`). The images will be converted and image patches are extracted. This step is required for efficient GPU execution. Choose patch sizes for your input dimensions `(Z)YX` and set how many patches should be extracted per image pair. After image patches have been extracted, they are saved to the output directory.

### Training the network
The training of a neural network is done iteratively in `epochs`. In each epoch, the network weights' are updated by optimizing a loss function on `steps_per_epoch` batches of image patches. The size of the batches is given by `batch_size`. To make use of all your image data, select `steps_per_epoch = #patches / batch_size`. Per default, 10% of patches are used for validation and not used in training. Select training parameters and execute training code block.

# CARE in command line

Training and predictions of CARE*less* CARE can also be done in the command line. You will need to have a CARE*less* settings file in .json format

```python
{
    "in_dir": "<path-to-input-images>",  # path to image pairs
    "out_dir": "<path-to-result-model>", # path to store CARE models 
    "low_wc": "*low*",    # a wild-card expression to select the low-quality images from in_dir 
    "high_wc": "*high*",  # a wild-card expression to select the high-quality images from in_dir 
    "axes": "ZYX",        # ZYX for 3D YX for 2D 
    "patch_size": [8, 64, 64],   # patch sizes
    "n_patches_per_image": 128,  # number of patches
    "train_channels": [0, 1],    # channels to train (and predict)
    "train_epochs": 7,           # training epochs
    "train_steps_per_epoch": 10, # training steps per epoch
    "train_batch_size": 32,      # training batch-size
    "probabilistic": false,      # probabilistic CARE?     
    "name": "my-care-test",      # choose model name 
    "low_scaling": [2.0, 1.992, 2.0] # the up-scaling (inferred automatically in training)
}
``` 
*careless_care.json*

### Training
To train type:

`careless_care_train --care_project careless_care.json`

### Prediction
To CARE-predict one or more low quality image type:

`careless_care_predict --care_project careless_care.json --ntiles 2 8 8 low_quality_new.tif <more-images>`

Due to memory GPU limitations CARE can be predicted tile-wise. The `ntiles` parameter sets the number of tiles per axis *ZYX* (default 1 4 4)

---


# Noise2Void in Jupyter

Noise2void does not require pairs of images. Just follow the steps in the notebook.

# Noise2Void in command line

Training and prediction with Noise2void can be done using the noise2void settings file in .json format

```python
{
    "name": "my-n2v-exp",         # N2Vmodel name 
    "in_dir": "<path-to-images>", # path to images
    "glob": "*.tif",              # wild-card selection
    "axes": "YX",                 # 2D (XY) or 3D (ZYX)
    "patch_size": [128, 128],     # path sizes
    "n_patches_per_image": -1,    # how many patches, use -1 for all
    "train_channels": [0, 1],     # channels to use
    "train_epochs": 40,           # training epochs
    "train_steps_per_epoch": 400, # training steps per epoch
    "train_batch_size": 16,       # training batch-size
    "n2v_perc_pix": 0.016,        # N2V default parameters 
    "n2v_patch_shape": [],        #  ~ 
    "n2v_neighborhood_radius": 5, #  ~
    "augment": true               # Augment training 8-fold by flips/rotations
}
```
*careless_n2v.json*

### Train and predict
To train and predict the images selected in `in_dir` type:

`careless_n2v --n2v_project careless_n2v.json --ntiles`

Optionally you can use the parameter `--model_name <my_name>` to overwrite the `name` given in the settings .json file

# Example data
The authors of [CARE](https://github.com/CSBDeep/CSBDeep/tree/master/examples) provide example data from different modalities.

* [3D denoising](http://csbdeep.bioimagecomputing.com/example_data/tribolium.zip)

Unzip, copy and rename (e. g. *_low.tif*, *_high.tif*) the images form `low` and `GT` into a single folder.

# Comparison of CARE with N2V
![CAREless user interface](img/example_result.png "Comparison of CARE with N2V")





