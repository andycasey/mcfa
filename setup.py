#!/usr/bin/env python

import os
from distutils.core import setup

version_file = os.path.join(os.path.dirname(__file__), "VERSION")
with open(version_file, "r") as fp:
    version = fp.read().strip()

setup(name="mcfa",
      version=version,
      description="Mixture of common factor analyzers",
      author="Andy Casey",
      author_email="andrew.casey@monash.edu",
      url="https://github.com/andycasey/mcfa",
      packages=["mcfa"],
      install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
      ])