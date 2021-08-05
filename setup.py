import os
from setuptools import find_packages, setup

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="radarqc",
    version="0.1.0",
    author="John Stanco",
    long_description=long_description,
    packages=find_packages(),
)
