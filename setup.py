#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import pathlib
import configparser

setup(name='kpicdrp',
      version='1.0',
      description='KPIC Data Reduction Pipeline',
      author='KPIC Team',
      packages=find_packages(),
     )

## Create a configuration file for the KPIC DRP if it doesn't exist. 

homedir = pathlib.Path.home()
config_filepath = os.path.join(homedir, ".kpicdrp")
if not os.path.exists(config_filepath):

      kpicdrp_basedir = os.path.dirname(__file__)

      config = configparser.ConfigParser()
      config.read(os.path.join(kpicdrp_basedir, ".kpicdrp_template"))
      config["PATH"]["caldb"] = kpicdrp_basedir
      with open(config_filepath, 'w') as f:
            config.write(f)

      print("kpicdrp: Configuration file written to {0}. Please edit if you want things stored in different locations.".format(config_filepath))


