import re
from setuptools import setup

with open("README.md", "rb") as f:
    description = f.read().decode("utf-8")

with open("careless/version.py", "r") as f:
    exec(f.read())

setup(
    name="CAREless",
    packages=["careless", "careless.care", "careless.n2v"],
    version=__version__,
    description=description,
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://git.ist.ac.at/csommer/careless",
    entry_points={
        "console_scripts": [
            "careless_n2v=careless.n2v.n2v:cmd_line",
            "careless_Care=careless.care.core:cmd_line",
        ]
    },
    author="Christoph Sommer",
    author_email="christoph.sommer23@gmail.com",
)

