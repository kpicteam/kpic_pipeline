import configparser
import os
import pathlib
import configparser

# load in default caldbs based on configuration file
config_filepath = os.path.join(pathlib.Path.home(), ".kpicdrp")
config = configparser.ConfigParser()
config.read(config_filepath)
datadir = config.get("PATH", "datadir", fallback=None)

if datadir is None:
    # use the default
    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
    

kpic_params = configparser.ConfigParser()
kpic_params.read(os.path.join(datadir, "kpic_params.ini"))