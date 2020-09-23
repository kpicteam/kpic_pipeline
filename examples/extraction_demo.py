
import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.table as table
import scipy.ndimage as ndi
import kpicdrp.extraction 
import os

# We will extract the flux for our RV calibrator, but in general this can be for anything

# load in the files we generated in the background_demo.py example 
# or we can load in the backup calibrations in the example data
# you can replace with your own generated backgrounds if you would like
mykpicdir = "../../public_kpic_data/"
sample_calib_dir = os.path.join(mykpicdir,"20200702_HIP_81497", "calib")
with fits.open(os.path.join(sample_calib_dir, "20200701_background_med_nobars_tint1.47528_coadds1.fits")) as hdulist:
    bkgd_1s = np.copy(hdulist[0].data)

with fits.open(os.path.join(sample_calib_dir, "20200701_persistent_badpix_nobars_tint1.47528_coadds1.fits")) as hdulist:
    badpixmap_1s = np.copy(hdulist[0].data)

# read in trace centers and trace widths
with fits.open(os.path.join(sample_calib_dir, "20200702_trace_loc_smooth.fits")) as hdulist:
    trace_centers = hdulist[0].data
with fits.open(os.path.join(sample_calib_dir, "20200702_line_width_smooth.fits")) as hdulist:
    trace_sigmas = hdulist[0].data

###### Fiber 2 ###########
data_dir = os.path.join(mykpicdir, "20200702_HIP_81497", "raw")
filestr = "nspec200702_0{0:03d}.fits"
filenums =  [2, 6]
filelist = [os.path.join(data_dir, filestr.format(i)) for i in filenums]

nobadpix = np.zeros(bkgd_1s.shape) # can be used if you want to skip bad pixel map application
sci_frame, sci_noise, sci_frames = kpicdrp.extraction.process_sci_raw2d(filelist, bkgd_1s, badpixmap_1s, detect_cosmics=False)

all_fluxes = []
all_errors = []
# extract fluxes from each frame
for frame, num in zip(sci_frames, filenums):
    fluxes, error = kpicdrp.extraction.extract_flux(sci_frame, "nspec200702_0{0:03d}_fluxes.fits".format(num), trace_centers, trace_sigmas, fit_background=True)

    all_fluxes.append(fluxes)
    all_errors.append(error)

# average the fluxes in time
fluxes = np.nanmedian(all_fluxes, axis=0)
errors = np.nanmedian(all_errors, axis=0)

# the star was on SF2, so let's get those specifically
sf2_fluxes = fluxes[1]
sf2_errors = errors[1]

import matplotlib.pylab as plt
fig = plt.figure(figsize=(10, 10))
i = 0
for flux_order, err_order in zip(sf2_fluxes, sf2_errors):
    i += 1
    fig.add_subplot(9, 1, i)
    xs = np.arange(flux_order.shape[0])
    plt.plot(xs, flux_order, 'b-', zorder=0)
    plt.fill_between(xs, flux_order - err_order, flux_order + err_order, color='c', zorder=-1)
    plt.ylim([0, np.nanpercentile(flux_order, 99) * 1.2])

plt.show()