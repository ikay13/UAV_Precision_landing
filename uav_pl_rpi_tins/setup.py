#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
     packages=['uav_pl'],
     package_dir={'': 'src/UAV_Precision_landing'}
)

setup(**setup_args)
