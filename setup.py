import re
from setuptools import setup
 
 
with open("README.md", "rb") as f:
    description = f.read().decode("utf-8")

setup(
    name = "bif_care",
    packages = ["bif_care"],
    version = "0.1",
    description = description,
    long_description = description,
    author = "Christoph Sommer",
    author_email = "christoph.sommer@gmail.com",
    )