# bif_care|n2v
Simple IPython based user interface to [CARE](http://csbdeep.bioimagecomputing.com/) a toolbox for Content-aware Image Restoration and to [Noise2void](https://github.com/juglab/n2v)

# care
## How to use:
CARE needs pairs of registered images - low (input) and high (output) quality. It trains a convolutional neural network how to transform low quality images - which might even be of less physical resolution - into high quality images. After training, newly recorded low quality images or movies can be predicted. 2D, 3D and multi-channel images are supported. For each channel a separate network is trained.

#### Vanilla screencast for input selection and training
![bif_care User interface](vid/bif_care_demo_01.mp4)

#### Input selection and patch extraction
0. Clone this repository with `git clone https://....`
1. Copy and rename the IPython notebook template file: `bif_care_templ.ipynb` to `my_care_project.ipynb`
2. Open your renamed `my_care_project.ipynb` file in Jypyter or IPyhton notebook.
3. In order to train CARE, the path to the image pairs needs to be specified. Then, choose images for low and high quality respectively using a wild-card (e.g. `low*.tif` and `high*.tif`). The images will be converted and image patches are extracted. This step is required for efficient GPU execution. Choose patch sizes for your input dimensions `(Z)YX` and set how many patches should be extracted per image pair. After image patches have been extracted, they are saved to the output directory. 

#### Training the network
The training of a neural network is done iteratively in `epochs`. In each epoch, the network weights' are updated by optimizing a loss function on `steps_per_epoch` batches of image patches. The size of the batches is given by `batch_size`. To make use of all your image data, select `steps_per_epoch = #patches / batch_size`. Per default, 10% of patches are used for validation and not used in training.

4. Select training parameters and execute training code block.

#### Seting up Fiji for prediction with CARE models
Using Tensorflow 1.12.0 / NVidia toolkit 9.0
1. Use the Fiji [CSBDeep](http://sites.imagej.net/CSBDeep/) update site

Using Tensorflow 1.13.1 / NVidia toolkit 10.0
1. get libtensorflow-1.13.1.jar and [tensorflow_jni.dll](https://www.tensorflow.org/install/lang_java)
2. put libtensorflow_jni-1.13.1.jar into "Fiji.app\jar\"
3. put tensorflow_jni.dll into folder "Fiji.app\lib\win64\"

Restart Fiji

#### Using the trained network for prediction
You can predict new images in the IPython notebook directly using the prediction widgets, or use the Fiji:

0. Add (or enable) [CSBDeep](http://sites.imagej.net/CSBDeep/) to your Fiji update sites
1. Open image you want to predict in Fiji
2. In Fiji choose `Plugins->CSBDeep->Run your network`
3. Select network file `<bif_care-out-folder>/models/CH_X_model/TF_SavedModel.zip` as 'Import model (.zip)' of your trained channel
4. Set additional parameters such as number of tiles (higher number, in case your entire image cannot be loaded on your GPU memory) and press OK

# Noise2void
## How to use:
Noise2void does not require pairs of images.
1. Copy and rename the IPython notebook template file: `bif_n2v_templ.ipynb` to `my_n2v_project.ipynb`
2. Open your renamed `my_n2v_project.ipynb` file in Jypyter or IPyhton notebook.
3. Follow steps in the notebook

# Installation (care and n2v)
### NVidia Cuda and cuDNN
1. Install [NVidia CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
2. Install [NVidia cuDNN](https://developer.nvidia.com/cudnn)

The installation of the CUDA toolkit is straight-forward. To download cuDNN you have to create a Developer account at NVidia. After downloading integrate the folders `bin`, `include` and `lib` into the according folders of your CUDA toolkit installation.

#### Known working versions
1. NVidia CUDA toolkit  9.0 + cuDNN 7.3.0 + tensorflow 12.0
2. NVidia CUDA toolkit 10.0 + cuDNN 7.5.1 + tensorflow 13.1 (recommended)

### Python dependencies
We strongly recommend using the [Anaconda Python distribution (64bit)](https://www.anaconda.com/distribution/) with Python == 3.6. Furthermore, we recommend to create a new conda virtual environment for Python 3.6 with:

```
conda create -n py36_bif_care python=3.6 anaconda
conda activate py36_bif_care
```

You can install all mayor dependencies, including tensforflow (1.12.1) and csbdepp (0.3.0) by:

```
cd <this-path>
pip install -r requirements.txt -e .
```
If you observe problems installing javabridge see troubleshooting below.

If your Python distribution comes without PyQt5, install it by:

```
conda install pyqt
```

### IPython widgets
The user interface is written using IPython/Jupyter widgets, which requires the installation of [node.js](https://nodejs.org/en/)).
Install it by using the official installer or by: `conda install -c conda-forge nodejs` from an Anaconda prompt.

Finally, you have to install jypyter widgets and enable them, by:

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter nbextension enable --py widgetsnbextension

```
More information on [Jupyter/IPython widgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html).

### Example data
The authors of [CARE](https://github.com/CSBDeep/CSBDeep/tree/master/examples) provide example data from different modalities.

* [3D denoising](http://csbdeep.bioimagecomputing.com/example_data/tribolium.zip)

Unzip, copy and rename (e. g. *_low.tif*, *_high.tif*) the images form `low` and `GT` into a single folder.

### Comparison of CARE with N2V
![bif_care User interface](img/example_result.png "Comparison of CARE with N2V")


### Troubleshooting and known issues
* tensorflow 1.13.x requires NVidia tookit 10.0 for the latest csbdeep 0.3.0 release. 
* Currently NVidia toolkit 10.1 is not supported by the latest tensorflow==13.1 release
* To install bioformats/ javabridge you need the Microsoft Visual Studio compiler runtime 2015 (14.0) installed, which is included with Microsoft Visual Studio community edition >=2017
* To install bioformats/ javabridge you need Java SDK 8 (1.8) or download [pre-compiled .whl](https://www.lfd.uci.edu/~gohlke/pythonlibs/) packages and install them by:

    ```pip install <javabridge_cp36_amd64>.whl` and `pip install <bioformatse_cp36_amd64>.whl```

* It may be required to launch Anaconda / Anaconda prompt as administrator





