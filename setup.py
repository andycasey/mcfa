#!/usr/bin/env python

import os
from setuptools import setup

version_path = os.path.join(os.path.dirname(__file__), "VERSION")
with open(version_path, "r") as fp:
    version = fp.read().strip()

requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
with open(requirements_path, "r") as fp:
    install_requires = fp.read().split("\n")


setup(name="mcfa",
      version=version,
      description="Mixture of common factor analyzers",
      author="Andy Casey",
      author_email="andrew.casey@monash.edu",
      url="https://github.com/andycasey/mcfa",
      packages=["mcfa"],
      install_requires=install_requires)