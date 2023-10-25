#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
globalHelioseismology - a Python package for processing global helioseismic time-series data
:copyright:
    Samarth G. Kashyap (samarth.g.kashyap@outlook.com), 2022
:license:
    MIT License
'''
# Importing setuptools monkeypatches some of distutils commands so things like
# 'python setup.py develop' work. Wrap in try/except so it is not an actual
# dependency. Inplace installation with pip works also without importing
# setuptools.

import os
import sys
import math
import argparse
from setuptools import setup
from setuptools import find_packages
from setuptools.command.test import test as TestCommand


setup(
    name='globalHelioseismology',
    version='1.0.0',
    packages=find_packages("src"), # Finds every folder with __init__.py in it. (Hehe)
    install_requires=[
        "numpy", "scipy"
    ],
)
