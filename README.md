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
Before running, you need to generate the following. Example scripts are provided in the examples folder.
1) calib_info file which specifies which frames are background, wavecal, and trace (calib_info.py)
2) Bad pixel maps and backgrounds (calc_background.py)
3) Trace file (calc_trace.py)

The nightly pipeline is called run_extraction_night.py. You run it with: 

    > python run_extraction_night.py
   
To run in real time (during observing), you need to constantly read the new files. Nodding is recommended since you might not have backgrounds. Run:

    > python run_extraction_night.py --nod_only=y --new_files=y --plot=y
     
