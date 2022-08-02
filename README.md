# KPIC Data Reduction Pipeline

## Install
Run in the top level of this repo:

    > pip install -r requirements.txt -e .

This will create a configuration file at `~/.kpicdrp` that will specify the path to where the KPIC DRP calibration databases live. By default, it is where the source code for the KPIC DRP lives. The following calibration databases will be defined:

  * caldb_detector.csv
  * caldb_traces.csv
  * caldb_wavecal.csv


### Pipeline to extract 1D spectrum for a night

## Pre-requisites: 
Before running, you need to have the
1) Trace file
2) Bad pixel maps and backgrounds 
for the night. This can be done with the relevant demo files in the examples folder.
3) Files for the night downloaded
4) Edit the following directories if not on hcig1: kpicdir, raw_datadir in the beginning of the script

The code is in the examples folder, with the name run_extraction_night.py. You run it with: 

    > python run_extraction_night.py
     
