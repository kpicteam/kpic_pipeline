# KPIC Data Reduction Pipeline

## Install
Run in the top level of this repo:

    > pip install -r requirements.txt -e .

This will create a configuration file at `~/.kpicdrp` that will specify the path to where the KPIC DRP calibration databases live. By default, it is where the source code for the KPIC DRP lives. The following calibration databases will be defined:

  * caldb_detector.csv
  * caldb_traces.csv
  * caldb_wavecal.csv

caldb_detector.csv defines the location of the thermal background as a function of exposure time as well as the bad pixel map. caldb_traces.csv defines the location of the trace normally using bright A0V stars. caldb_wavecal.csv defines the wavelength solution, typically using bright early M giant star spectra. Note that when you rerun the trace or wavelength calibration using a different target or model, you will need to remove the earlier entry to ensure the most recent files are properly used in the following steps.

To make use of our wavelength calibration routine, you will need to download initial guesses from our [Public KPIC Data](https://drive.google.com/drive/folders/1eE3z_tbaXViGJCNN3dJswu4HeYBTa0hl) if you have not reduced any KPIC data yet. There are also examples of raw and reduced KPIC spectra on the drive for you to compare.

### Pipeline to extract 1D spectrum for a night

## Pipeline Instructions to Reduce Nightly Data: 
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

## Background/Bad Pixel Finding 

Select a location on your machine to store, process, and reduce data
* Output locations are flexible in data reduction scripts
* Identify which nspec files are Background exposures
* Make a folder and copy valid background files for the night
```
cd /mykpicdatadir/
```
```
mkdir calibs;cd calibs
```
```
mkdir 20221009;cd 20221009 (replace with date of your data)
```
```
mkdir bkgd_bdmap;cd bkgd_bdmap
```
```
mkdir raw
```

* Move the pertinent files to this directory 
* Make a copy of the [background_demo.py](https://github.com/kpicteam/kpic_pipeline/blob/mod_readme/examples/background_demo.py) script from /kpic_pipeline/examples and change the paths to match where the data is  

<img width="800" alt="unknown" src="https://github.com/kpicteam/kpic_pipeline/assets/74935396/9fd4637e-d60d-419f-9117-0d60d4648cc3">
 
Output should give files like: [bkgd_fileprefix]_background_med_nobars_tint[integration_time]_coadds1.fits and [bkgd_fileprefix] _persistent_badpix_nobars_tint[integration_time]_coadds1.fits for each true integration time (tint) used  

Result should look like ￼
 ![1__#$!@%!#__unknown](https://github.com/kpicteam/kpic_pipeline/assets/74935396/bb84e57c-702f-4b6c-af61-b4f0e3b275da)


  
## Trace Finding 
* Identify which nspec files are telluric calibrator (A0V star) exposures
* Make a folder for trace and copy valid data files
```
cd ~/mykpicdatadir/20221009/  (replace with your path and date)
```
```
mkdir trace  
```
```
mkdir raw
```

* Make a copy of the [trace_demo.py](https://github.com/kpicteam/kpic_pipeline/blob/mod_readme/examples/trace_demo.py) script from /kpic_pipeline/examples and change the paths to match where the data is

<img width="902" alt="Screenshot 2024-05-13 at 11 02 16 AM" src="https://github.com/kpicteam/kpic_pipeline/assets/74935396/f8b3845d-cc3a-4c38-9747-2be4b4ca698c">

  
* Output should give files like: nspec20200928_[last_filenumber]_trace.fits 
* Output should look something like: ￼

![3__#$!@%!#__unknown](https://github.com/kpicteam/kpic_pipeline/assets/74935396/fd11045c-0f57-426c-9902-6eb790156c0c)

The star fibers 1-4 are labeled in s1, s2, s3, s4, respectively, whereas the background/dark traces are in between these fibers are labeled as b0, b1, b2, b3, or d0, d1, d2, d3.

## Flux Extraction 
* Start with the wavelength calibrator/RV standard (e.g. HIP 81497 on 20220721) 
* Find/Make a KPIC Campaign Science folder for the RV standard and copy valid data files
```  
cd ~/mykpicdatadir
```
```
mkdir science;cd science
```
```
mkdir HIP81497 (replace target name with your wavelength calibrator or RV standard)
```
```
mkdir 20220721;cd 20220721 (swap date with your observation date)
```
```
mkdir raw
```

* Make a copy of the [extraction_demo.py](https://github.com/kpicteam/kpic_pipeline/blob/mod_readme/examples/extraction_demo.py) script from /kpic_pipeline/examples
* There are a few options to note in this script:
    * A choice of a box extraction or optimal extraction
    * A choice of the background subtraction method, either using the main background file or using a nod subtraction if your dataset uses fiber bouncing during observation
```python
# define the extraction methods
box = False # box or optimal extraction
```

```python
subtract_method = 'bkgd' # bkgd (background) or nod (nod subtraction/pair subtraction)
```
* and change the paths to match where the data is￼  

<img width="746" alt="Screenshot 2024-05-13 at 11 14 50 AM" src="https://github.com/kpicteam/kpic_pipeline/assets/74935396/2427b7ab-d3db-47aa-9428-b6dba64321d2">


* Output should give files like: nspec220721_[filenumbers]_bkgdsub_spectra.fits 

* Output should look something like, though it will depend on your exposure time and throughput for the night: ￼

![5__#$!@%!#__unknown](https://github.com/kpicteam/kpic_pipeline/assets/74935396/2aab7d86-6f90-41cd-8763-bddb828a838e)


*To check throughput on calibrators, modify and run kpic_pipeline/examples/throughput_demo.py   


## Wavelength Calibration 

KPIC uses a bright early M giant star which utilizes CO lines as well as earth telluric absorption to extract the wavelength solutions. The three best orders in K band for science analyses are orders 31-33 (2.29-2.49 microns), but KPIC wavelength calibration attempts to provide the wavelength solutions for all nine NIRSPEC orders 31-39. To enable this routine, make sure that you have downloaded the initial guesses and earth atmosphere models from the [utils](https://drive.google.com/drive/folders/1wKh21-kfQ4l7wPW_muy1ikWvgigWKRxV) folder in the [Public KPIC Data](https://drive.google.com/drive/folders/1eE3z_tbaXViGJCNN3dJswu4HeYBTa0hl). The initial guesses of the wavelength solutions are 'first_guess_wvs_20200607_HIP_81497.fits' or 'first_guess_wvs_20200928_HIP_81497.fits'. To model the M giant stellar atmosphere, the KPIC DRP uses PHOENIX-ACES-AGSS-COND-2011 models at a given effective and surface gravity (e.g. HIP81497_lte03600-1.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits) and requires an input radial velocity for the M giant spectra.

* Make a calib folder for wavelength calibration in KPIC_Campaign
```
cd /mykpicdatadir/calibs/20220721 (replace with your date and directory)
```
```
mkdir wave;cd wave
```

* Make a copy of the [wavecal_demo.py](https://github.com/kpicteam/kpic_pipeline/blob/mod_readme/examples/wavcal_demo.py) script from /kpic_pipeline/examples and change the paths to match where the data is (Be careful here, there is a lot to change) 

 <img width="1000" alt="6__#$!@%!#__unknown" src="https://github.com/kpicteam/kpic_pipeline/assets/74935396/a9af6655-cf43-4e52-8fcf-fe95bb42e271">
 

* Be sure to change the RV value to match the wavecal star you are using￼  

 <img width="800" alt="7__#$!@%!#__unknown" src="https://github.com/kpicteam/kpic_pipeline/assets/74935396/f2fca401-30c9-4a10-bed2-2cdcfbccc075">
 
If the wavelength solution is bad, you can try to use another initial wavelength solution file as a starting point, or you can widen the grid search delta wavelength (grid_dwv) from 1e-4 to 3e-4 microns. These should in principle provide a reasonable wavelength calibration.

* File outputs should look like: 20220721_HIP81497_psg_wvs.fits  
* The script will plot images for each fiber to show how the wavelength solution fits and save the plots; make sure these are in the wave directory as well 
* Solution should look something like this for each fiber:
  
 ![8__#$!@%!#__unknown](https://github.com/kpicteam/kpic_pipeline/assets/74935396/2d0a73a5-f5e3-40bc-8ae5-2245cc4f178c)

￼
## Rinse and Repeat Flux Extraction for all Host/Companions/Targets that you wish to analyze 
 
