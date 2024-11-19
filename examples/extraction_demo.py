import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.table as table
import scipy.ndimage as ndi
import os
import multiprocessing as mp
from glob import glob
import kpicdrp.utils as utils
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import kpicdrp.trace as trace
import kpicdrp.extraction as extraction
import kpicdrp.data as data
from kpicdrp.caldb import det_caldb, trace_caldb

# We will extract the flux for our RV calibrator, but in general this can be for anything

# load in the files we generated in the background_demo.py example
# or we can load in the backup calibrations in the example data
# you can replace with your own generated backgrounds if you would like

mypool = mp.Pool(2)

# define the extraction methods
box = False # box or optimal extraction
subtract_method = 'bkgd' # bkgd (background) or nod (nod subtraction/pair subtraction)

# Public KPIC google drive
kpicpublicdir = "." # main data dir

raw_data_dir = os.path.join(kpicpublicdir,"20200928_HIP_95771", "raw") # raw 2D images
out_flux_dir = os.path.join(kpicpublicdir,"20200928_HIP_95771", "fluxes")
# raw_data_dir = os.path.join(kpicpublicdir,"20200928_zet_Aql", "raw") # raw 2D images
# out_flux_dir = os.path.join(kpicpublicdir,"20200928_zet_Aql", "fluxes")
# raw_data_dir = os.path.join(kpicpublicdir,"20200928_HR_7672", "raw") # raw 2D images
# out_flux_dir = os.path.join(kpicpublicdir,"20200928_HR_7672", "fluxes")
# raw_data_dir = os.path.join(kpicpublicdir,"20200928_HR_7672_B", "raw") # raw 2D images
# out_flux_dir = os.path.join(kpicpublicdir,"20200928_HR_7672_B", "fluxes")
if not os.path.exists(os.path.join(out_flux_dir)):
    os.makedirs(os.path.join(out_flux_dir))
filelist = glob(os.path.join(raw_data_dir, "*.fits"))
filelist.sort()

raw_sci_dataset = data.Dataset(filelist=filelist, dtype=data.DetectorFrame)

# only nod/pair subtraction requires fiber information; this only works for data older than 2021 October
if subtract_method == 'nod':
    fiber_goals = raw_sci_dataset.get_header_values("FIUGNM")
    fiber_goals = [int(i[-1]) for i in fiber_goals]
    print('fiber_goals', fiber_goals)

# fetch calibration files
bkgd = det_caldb.get_calib(raw_sci_dataset[0], type="Background")
badpixmap = det_caldb.get_calib(raw_sci_dataset[0], type="BadPixelMap")

trace_dat = trace_caldb.get_calib(raw_sci_dataset[0])

# get background traces if they aren't there already
if 'b1' not in trace_dat.labels:
    trace_dat = trace.get_background_traces(trace_dat)

if subtract_method == 'bkgd':
    sci_dataset = extraction.process_sci_raw2d(raw_sci_dataset, bkgd, badpixmap, detect_cosmics=True, add_baryrv=True)

    spectral_dataset = extraction.extract_flux(sci_dataset, trace_dat, fit_background=True, bad_pixel_fraction=0.01, pool=mypool, box=box)
elif subtract_method == 'nod':
    sci_dataset = extraction.process_sci_raw2d(raw_sci_dataset, None, badpixmap, detect_cosmics=True, add_baryrv=True, nod_subtraction='nod', fiber_goals=fiber_goals)

    spectral_dataset = extraction.extract_flux(sci_dataset, trace_dat, fit_background=False, bad_pixel_fraction=0.01, pool=mypool, box=box)


spectral_dataset.save(filedir=out_flux_dir)

for filename in filelist:
    out_filename = os.path.join(out_flux_dir, os.path.basename(filename).replace(".fits", f"_{subtract_method}sub_spectra.fits"))
    this_spectrum = data.Spectrum(filepath=out_filename)
    spec = this_spectrum.fluxes
    err = this_spectrum.errs
    for fib_id in trace_dat.get_sci_indices():
        fig = plt.figure(fib_id + 1, figsize=(10, 10))
        for order_id in range(spec.shape[1]):
            plt.subplot(spec.shape[1], 1, order_id + 1)
            xs = np.arange(spec[fib_id, order_id, :].shape[0])
            plt.plot(xs, spec[fib_id, order_id, :], label=os.path.basename(filename).replace(".fits", ""))
plt.show()
