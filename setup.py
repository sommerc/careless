import re
from setuptools import setup

with open("README.md", "rb") as f:
    description = f.read().decode("utf-8")

with open('careless/version.py', "r") as f:
    exec(f.read())

setup(
    name = "CAREless",
    packages = ["careless"],
    version = __version__,
    description = description,
    long_description = description,
    long_description_content_type='text/markdown',
    url='https://git.ist.ac.at/csommer/careless',
    entry_points = {'console_scripts': [
            'careless_n2v=careless.n2v.n2v:cmd_line',
            'careless_Care=careless.care.core:cmd_line',
        ]},
    author = "Christoph Sommer",
    author_email = "christoph.sommer23@gmail.com",
    install_requires=[
            f"csbdeep=={__care_version__}",
            f"n2v=={__n2v_version__}",
            "scikit_image>=0.14.2",
            "tifffile>=2019.3.18",
            "widgetsnbextension>=3.4.2",
            "ipywidgets>=7.4.2",
            "javabridge>=1.0.18",
            "python-bioformats>=1.5.2",
            "jupyterlab>=0.35.6",
            "tensorflow-gpu==1.13.1,<2.0.0",
            "h5py<2.10.0",
            "scipy>=1.3.1",
            "keras<2.3.0"
      ]
    )