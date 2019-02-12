# bif_care
Simple ipython based interface to [CARE](http://csbdeep.bioimagecomputing.com/) a toolbox for Content-aware Image Restoration.
## How to use
Care needs paris of images - low and high quality. It learns how to convert low quality images (these can be of less resolution) into high quality images. After training, newly recorded low quality images (e.g. longer time-courses) can be predicted.

## Base requirments
* tensorflow
* tqdm (conda)
* bioformats (Gohlke)
* javabridge (Gohlke)
* tifffile (Gohlke)
* PyQt5 (conda)
* [nodejs](https://nodejs.org/en/)
* [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html) (conda + enabling)


