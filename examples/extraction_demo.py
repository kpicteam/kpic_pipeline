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

# Public KPIC google drive
kpicpublicdir = "fill/in/your/path/public_kpic_data/" # main data dir

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


raw_sci_dataset = data.Dataset(filelist=filelist, dtype=data.DetectorFrame)

# fetch calibration files
bkgd = det_caldb.get_calib(raw_sci_dataset[0], type="Background")
badpixmap = det_caldb.get_calib(raw_sci_dataset[0], type="BadPixelMap")

trace_dat = data.TraceParams(filepath=trace_caldb.db['Filepath'][0])

# get background traces if they aren't there already
if 'b1' not in trace_dat.labels:
    trace_dat = trace.get_background_traces(trace_dat)

sci_dataset = extraction.process_sci_raw2d(raw_sci_dataset, bkgd, badpixmap, detect_cosmics=True, add_baryrv=True)

spectral_dataset = extraction.extract_flux(sci_dataset, trace_dat, fit_background=True, bad_pixel_fraction=0.01, pool=mypool)

spectral_dataset.save(filedir=out_flux_dir)

for filename in filelist:
    out_filename = os.path.join(out_flux_dir, os.path.basename(filename).replace(".fits", "_bkgdsub_spectra.fits"))
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
