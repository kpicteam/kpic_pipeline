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
     
### Pipeline Instructions to Reduce Individual Datasets
## DRP Steps  

Outline 
* Background/Bad Pixel Finding 
* Telluric/RV Trace Finding 
* RV Standard Flux Extraction 
* RV Standard Wavelength Calibration 
* Target Flux Extraction 
* Cross Correlation  

## Background/Bad Pixel Finding 

Select a location on your machine to store, process, and reduce data
* Output locations are flexible in data reduction scripts
* Identify which nspec files are Background exposures via the nightly log 
* Make a folder and copy valid background files for the night   
    > cd /mykpicdatadir/
	> mkdir calibs;cd calibs  
	> mkdir 20221009;cd 20221009 (replace with date of your data)
	> mkdir bkgd_bdmap;cd bkgd_bdmap 
	> mkdir raw 
* Move the pertinent files to this directory 
* Make a copy of the background_demo.py script from /kpic_pipeline/examples and change the paths to match where the data is  
		
￼

 
	•	 Output should give files like: [bkgd_fileprefix]_background_med_nobars_tint1.47528_coadds1.fits and [bkgd_fileprefix] _persistent_badpix_nobars_tint1.47528_coadds1.fits for each tint used  
	•	
	•	Result should look like ￼
  
Trace Finding 
	•	Identify which nspec files are telluric calibrator exposures via nightly log 
	•	Make a folder for trace and copy valid data files  
	⁃	cd ~/mykpicdatadir/date/  (replace with your specific path)
	⁃	mkdir trace  
	⁃	mkdir raw  
	•	Make a copy of the trace_demo.py script from /kpic_pipeline/examples and change the paths to match where the data is

	•	￼  
	•	Output should give files like: nspec221009_[filenumber]_trace.fits 
	•	Output should look something like: ￼ 

Flux Extraction 
	⁃	Start with the wavelength calibrator/RV standard (e.g. HIP 81497 on 20220721) 
	⁃	Find/Make a KPIC Campaign Science folder for the RV standard and copy valid data files  
	⁃	cd ~/mykpicdatadir/
	⁃	mkdir science
	⁃	mkdir HIP81497 (e.g.)  
	⁃	mkdir 20220721 (replace with your date) 
	⁃	mkdir raw  
	⁃	Make a copy of the extraction_demo.py script from /kpic_pipeline/examples and change the paths to match where the data is￼  
Output should give files like: nspec220721_[filenumbers]_bkgdsub_spectra.fits 
Output should look something like, though it will depend on your exposure time and throughput for the night: ￼
To check throughput on calibrators, modify and run kpic_pipeline/examples/throughput_demo.py   
Wavelength Calibration 
	•	Make a calib folder for wavelength calibration in KPIC_Campaign 
	⁃	cd /mykpicdatadir/calibs/20220721 (replace with your date and directory)  
	⁃	mkdir wave  
	•	Make a copy of the wavecal_demo.py script from /kpic_pipeline/examples and change the paths to match where the data is (Be careful here, there is a lot to change) 
	•	￼  
	•	Be sure to change the RV value to match the wavecal star you are using￼  
	•	File outputs should look like: 20220721_HIP81497_psg_wvs.fits  
	•	The script will plot images for each fiber to show how the wavelength solution fits and save the plots; make sure these are in the wave directory as well 
	•	Solution should look something like this for each fiber: 
￼

 
Rinse and Repeat Flux Extraction for all Host/Companions/Targets that you wish to analyze 
 
