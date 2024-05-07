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

     
## Pipeline Instructions to Reduce Individual Datasets
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

 
Output should give files like: [bkgd_fileprefix]_background_med_nobars_tint1.47528_coadds1.fits and [bkgd_fileprefix] _persistent_badpix_nobars_tint1.47528_coadds1.fits for each tint used  

<img width="800" alt="unknown" src="https://github.com/kpicteam/kpic_pipeline/assets/74935396/9fd4637e-d60d-419f-9117-0d60d4648cc3">

Result should look like ￼
 ![1__#$!@%!#__unknown](https://github.com/kpicteam/kpic_pipeline/assets/74935396/bb84e57c-702f-4b6c-af61-b4f0e3b275da)


  
## Trace Finding 
* Identify which nspec files are telluric calibrator exposures via nightly log 
* Make a folder for trace and copy valid data files  
>	cd ~/mykpicdatadir/date/  (replace with your specific path)

>	mkdir trace  

>	mkdir raw


* Make a copy of the trace_demo.py script from /kpic_pipeline/examples and change the paths to match where the data is

  <img width="800" alt="2__#$!@%!#__unknown" src="https://github.com/kpicteam/kpic_pipeline/assets/74935396/bc250661-eb85-4899-a317-a9df8ef06b4f">
  
* Output should give files like: nspec221009_[filenumber]_trace.fits 
* Output should look something like: ￼ 
![3__#$!@%!#__unknown](https://github.com/kpicteam/kpic_pipeline/assets/74935396/fd11045c-0f57-426c-9902-6eb790156c0c)

## Flux Extraction 
* Start with the wavelength calibrator/RV standard (e.g. HIP 81497 on 20220721) 
* Find/Make a KPIC Campaign Science folder for the RV standard and copy valid data files
  
> cd ~/mykpicdatadir

> mkdir science;cd science

> mkdir HIP81497 (replace target name with your wavelength calibrator or RV standard)

> mkdir 20220721;cd 20220721 (swap date with your observation date)

> mkdir raw


* Make a copy of the extraction_demo.py script from /kpic_pipeline/examples and change the paths to match where the data is￼  

 <img width="787" alt="4__#$!@%!#__unknown" src="https://github.com/kpicteam/kpic_pipeline/assets/74935396/3c507353-2dc2-4826-879b-a53dec95f46a">

* Output should give files like: nspec220721_[filenumbers]_bkgdsub_spectra.fits 

* Output should look something like, though it will depend on your exposure time and throughput for the night: ￼

![5__#$!@%!#__unknown](https://github.com/kpicteam/kpic_pipeline/assets/74935396/2aab7d86-6f90-41cd-8763-bddb828a838e)


*To check throughput on calibrators, modify and run kpic_pipeline/examples/throughput_demo.py   


## Wavelength Calibration 
* Make a calib folder for wavelength calibration in KPIC_Campaign 
> cd /mykpicdatadir/calibs/20220721 (replace with your date and directory)  
> mkdir wave;cd wave  

* Make a copy of the wavecal_demo.py script from /kpic_pipeline/examples and change the paths to match where the data is (Be careful here, there is a lot to change) 

 <img width="800" alt="6__#$!@%!#__unknown" src="https://github.com/kpicteam/kpic_pipeline/assets/74935396/a9af6655-cf43-4e52-8fcf-fe95bb42e271">
 

* Be sure to change the RV value to match the wavecal star you are using￼  

 <img width="800" alt="7__#$!@%!#__unknown" src="https://github.com/kpicteam/kpic_pipeline/assets/74935396/f2fca401-30c9-4a10-bed2-2cdcfbccc075">
 

* File outputs should look like: 20220721_HIP81497_psg_wvs.fits  
* The script will plot images for each fiber to show how the wavelength solution fits and save the plots; make sure these are in the wave directory as well 
* Solution should look something like this for each fiber:
  
 ![8__#$!@%!#__unknown](https://github.com/kpicteam/kpic_pipeline/assets/74935396/2d0a73a5-f5e3-40bc-8ae5-2245cc4f178c)

￼
## Rinse and Repeat Flux Extraction for all Host/Companions/Targets that you wish to analyze 
 
