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


# We will extract the flux for our RV calibrator, but in general this can be for anything

# load in the files we generated in the background_demo.py example
# or we can load in the backup calibrations in the example data
# you can replace with your own generated backgrounds if you would like

mypool = mp.Pool(2)

# Public KPIC google drive
kpicpublicdir = "fill/in/your/path/public_kpic_data/" # main data dir
kpicpublicdir = "../public_kpic_data"

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

# master background and bad pix directory.
mybkgdir = os.path.join(kpicpublicdir,"20200928_backgrounds","calib")

background_med_filename = os.path.join(mybkgdir,"20200928_background_med_nobars_tint4.42584_coadds1.fits")
persisbadpixmap_filename = os.path.join(mybkgdir,"20200928_persistent_badpix_nobars_tint4.42584_coadds1.fits")

mytrfilename = os.path.join(kpicpublicdir,"20200928_zet_Aql","calib","nspec200928_0049_trace.fits")

# read the master Background file
bkgd = data.Background(filepath=background_med_filename)
# read the bad pixel map
badpixmap = data.BadPixelMap(filepath=persisbadpixmap_filename)

trace_dat = data.TraceParams(filepath=mytrfilename)

traces_with_bkgds = trace.get_background_traces(trace_dat)

# trace_loc_wbkg = np.concatenate([trace_loc, trace_loc_slit, trace_loc_dark], axis=0)
# trace_sigmas_wbkg = np.concatenate([trace_sigmas, trace_sigmas, trace_sigmas], axis=0)
# trace_flags = np.array([0, ] * trace_loc.shape[0] + [1, ] * trace_loc.shape[0] + [2, ] * trace_loc.shape[0])

raw_sci_dataset = data.Dataset(filelist=filelist, dtype=data.DetectorFrame)
sci_dataset = extraction.process_sci_raw2d(filelist, bkgd, badpixmap, detect_cosmics=True, scale=False, add_baryrv=False)

for filename,dat,header in zip(filelist,sci_frames,sci_hdrs):
    out_filename = os.path.join(out_flux_dir, os.path.basename(filename).replace(".fits", "_fluxes.fits"))
    fluxes, errors_extraction, errors_bkgd_only = extraction.extract_flux(dat, trace_loc_wbkg, trace_sigmas_wbkg,
                                                                            output_filename=out_filename,
                                                                            img_hdr=header,
                                                                            img_noise=bkgd_noise, fit_background=True,
                                                                            trace_flags=trace_flags, pool=mypool,
                                                                            bad_pixel_fraction=0.01)

    # out_hdr = extraction.add_baryrv(header)
    # prihdu = fits.PrimaryHDU(data=fluxes[np.where(trace_flags==0)], header=out_hdr)
    # exthdu1 = fits.ImageHDU(data=errors_extraction[np.where(trace_flags==0)],)
    # exthdu2 = fits.ImageHDU(data=fluxes[np.where(trace_flags==1)],)
    # exthdu3 = fits.ImageHDU(data=fluxes[np.where(trace_flags==2)],)
    # exthdu4 = fits.ImageHDU(data=errors_bkgd_only[np.where(trace_flags==0)],)
    # hdulist = fits.HDUList([prihdu, exthdu1, exthdu2, exthdu3,exthdu4])
    # hdulist.writeto(out_filename, overwrite=True)

for filename in filelist:
    out_filename = os.path.join(out_flux_dir, os.path.basename(filename).replace(".fits", "_fluxes.fits"))
    hdulist = fits.open(out_filename)
    spec = hdulist[0].data
    err = hdulist[1].data
    spec_slit = hdulist[2].data
    spec_dark = hdulist[3].data
    err_bkg = hdulist[4].data
    for fib_id in range(spec.shape[0]):
        fig = plt.figure(fib_id + 1, figsize=(10, 10))
        for order_id in range(spec.shape[1]):
            plt.subplot(spec.shape[1], 1, order_id + 1)
            xs = np.arange(spec[fib_id, order_id, :].shape[0])
            plt.plot(xs, spec[fib_id, order_id, :], label=os.path.basename(filename).replace(".fits", ""))
            # plt.plot(xs, spec_slit[fib_id,order_id,:])
            # plt.plot(xs, spec_dark[fib_id,order_id,:])
            # plt.fill_between(xs, spec[fib_id,order_id,:] - err[fib_id,order_id,:], spec[fib_id,order_id,:] + err[fib_id,order_id,:], zorder=-1)
            # plt.fill_between(xs, spec_slit[fib_id,order_id,:] - err_bkg[fib_id,order_id,:], spec_slit[fib_id,order_id,:] + err_bkg[fib_id,order_id,:], zorder=-1)
plt.show()

#
#     ##
#
# nobadpix = np.zeros(bkgd.shape) # can be used if you want to skip bad pixel map application
# mean_sci_data, sci_hdrs, sci_noise, sci_frames = kpicdrp.extraction.process_sci_raw2d(filelist, bkgd,
#                                                         badpixmap, detect_cosmics=False)
#
# # extract fluxes frame by frame, and save output from each frame
# all_fluxes = []
# all_errors = []
# # option of multiprocessing, arg is num processes - set pool to None if not desired
# pool = mp.Pool(2)
#
# for frame, num, hdr in zip(sci_frames, filenums, sci_hdrs):
#     fluxes, error = kpicdrp.extraction.extract_flux(frame,
#                                 os.path.join(target_dir + "nspec200702_0{0:03d}_fluxes.fits".format(num)),
#                                 trace_centers, trace_sigmas, img_hdr=hdr, fit_background=True, pool=pool)
#
#     all_fluxes.append(fluxes)
#     all_errors.append(error)
#
# # average the fluxes in time (as a quick check)
# fluxes = np.nanmedian(all_fluxes, axis=0)
# errors = np.nanmedian(all_errors, axis=0)
#
# # plot fluxes from a specific fiber
# fib_ind = 1  # i.e. fiber 2, where science target is in this case
# sf2_fluxes = fluxes[fib_ind]
# sf2_errors = errors[fib_ind]
#
# import matplotlib.pylab as plt
# fig = plt.figure(figsize=(10, 10))
# i = 0
# for flux_order, err_order in zip(sf2_fluxes, sf2_errors):
#     i += 1
#     fig.add_subplot(9, 1, i)
#     xs = np.arange(flux_order.shape[0])
#     plt.plot(xs, flux_order, 'b-', zorder=0)
#     plt.fill_between(xs, flux_order - err_order, flux_order + err_order, color='c', zorder=-1)
#     plt.ylim([0, np.nanpercentile(flux_order, 99) * 1.2])
#
# plt.show()