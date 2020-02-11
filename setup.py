import re
from setuptools import setup

with open("README.md", "rb") as f:
    description = f.read().decode("utf-8")

setup(
    name = "CAREless",
    packages = ["careless"],
    version = "0.3",
    description = description,
    long_description = description,
    entry_points = {'console_scripts': ['careless_n2v=careless.n2v.n2v:cmd_line',
                                         'careless_Care=careless.care.core:cmd_line',
                                            ]},
    author = "Christoph Sommer",
    author_email = "christoph.sommer23@gmail.com",
    )