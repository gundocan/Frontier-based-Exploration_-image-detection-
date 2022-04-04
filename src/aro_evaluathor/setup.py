#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_kwargs = generate_distutils_setup()
setup_kwargs['packages'] = ['aro_evaluathor']
setup_kwargs['package_dir'] = {'': 'src'}

setup(**setup_kwargs)
