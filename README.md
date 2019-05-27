# bif_care
Simple IPython based user interface to [CARE](http://csbdeep.bioimagecomputing.com/) a toolbox for Content-aware Image Restoration.
## How to use:
CARE needs pairs of registered images - low (input) and high (output) quality. It trains a convolutional neural network how to transform low quality images - which might even be of less physical resolution - into high quality images. After training, newly recorded low quality images or movies can be predicted. 2D, 3D and multi-channel images are supported. For each channel a separate network is trained.

#### Input selection and patch extraction
In order to train CARE the path to the image pairs needs to be specified. Then, choose images for low and high quality respectively using a wild-card (e.g. `low*.tif` and `high*.tif`). The images will be converted and image patches are extracted. This step is required for efficient GPU execution. Choose patch sizes for your input dimensions `(Z)YX` and set how many patches should be extracted per image pair. After image patches have been extracted, they are saved to the output directory. 

#### Training the network
The training of a neural network is done iteratively in `epochs`. In each epoch the network weights' are updated by optimizing a loss function of `steps_per_epoch` batches of image patches. The size of this batches are given by `batch_size`. To make use of all your image data, select `steps_per_epoch = #patches / batch_szie`. Per default, 10% of patches are used for validation and not used in training.

#### Vanilla screencast for input selection and training
![bif_care User interface](vid/bif_care_demo_01.mp4)

## Installation

### NVidia Cuda and cuDNN

### Python dependencies

### IPython widgets




## Base requirments
* tensorflow
* tqdm (conda)
* bioformats (Gohlke)
* javabridge (Gohlke) 
* tifffile (Gohlke) 
* PyQt5 (conda) 
* [nodejs](https://nodejs.org/en/)
* [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html) (conda + enabling)


