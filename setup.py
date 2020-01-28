import re
from setuptools import setup
 
with open("README.md", "rb") as f:
    description = f.read().decode("utf-8")

setup(
    name = "bif_care",
    packages = ["bif_care"],
    version = "0.2",
    description = description,
    long_description = description,
    entry_points = {'console_scripts': ['bif_n2v=bif_n2v.bif_n2v:cmd_line']},
    author = "Christoph Sommer",
    author_email = "christoph.sommer23@gmail.com",
    )