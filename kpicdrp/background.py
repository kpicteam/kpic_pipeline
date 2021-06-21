import os
import warnings
import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import interp1d
from scipy.ndimage.filters import convolve
from astropy.io import fits
import astropy.time as time
from astropy.stats import mad_std

import kpicdrp.data as data

import logging
import matplotlib.pyplot as plt

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def get_badpixmap_from_laplacian(image,bad_pixel_fraction=1e-2):
    med_val = np.nanmedian(image)
    mad_val = mad_std(image[np.where(np.isfinite(image))])
    laplacian = np.zeros((3, 3))
    laplacian[1, 1] = -4
    laplacian[0, 1], laplacian[1, 0], laplacian[1, 2], laplacian[2, 1] = 1, 1, 1, 1

    laplacian_map = convolve(image, laplacian, mode="constant", cval=0, origin=[0, 0])

    hist, bin_edges = np.histogram(laplacian_map, bins=np.linspace(-100 * mad_val + med_val, 100 * mad_val + med_val, 200 * 10))
    bin_center = (bin_edges[1::] + bin_edges[0:len(bin_edges) - 1]) / 2.
    ind = np.argsort(hist)
    cum_posterior = np.zeros(np.shape(hist))
    cum_posterior[ind] = np.cumsum(hist[ind])
    cum_posterior = cum_posterior / np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    if len(bin_center[0:argmax_post]) < 2:
        lower_bound = np.nan
    else:
        lf = interp1d(cum_posterior[0:argmax_post], bin_center[0:argmax_post], bounds_error=False, fill_value=np.nan)
        lower_bound = lf(bad_pixel_fraction)
    if len(bin_center[argmax_post::]) < 2:
        upper_bound = np.nan
    else:
        rf = interp1d(cum_posterior[argmax_post::], bin_center[argmax_post::], bounds_error=False, fill_value=np.nan)
        upper_bound = rf(bad_pixel_fraction)

    where_bad_pixels = np.where((laplacian_map < lower_bound) + (laplacian_map > upper_bound))

    bad_pix_map = np.ones(image.shape)
    bad_pix_map[where_bad_pixels] = np.nan

    return bad_pix_map

def get_badpixmap_from_mad(image,threshold=100):
    bad_pix_map = np.ones(image.shape)
    image_mad = mad_std(image[np.where(np.isfinite(image))])
    bad_pix_map[np.where(np.abs(image-np.nanmedian(image))>threshold*image_mad)] = np.nan
    return bad_pix_map

def get_badpixmap_from_readnoisebars(image,header):
    bad_pix_map = np.ones(image.shape)
    if "MCDS" == header["SAMPMODE"].strip():
        for col in np.arange(129, 2048, 128):
            bad_pix_map[col - 1:col + 2, :] = np.nan
    if "CDS" == header["SAMPMODE"].strip():
        for col in np.arange(129 - 64-1, 2048, 64):
            bad_pix_map[col, :] = np.nan

    return bad_pix_map

def create_background_badpixelmap(background_frames, fileprefix=None, plot=False):
    """
    Creates a background and backpixelmap based on input thermal background calibration data

    Args:
        background_files (data.Dataset): Dataset of DetectorFrames corresponding to the raw frames of a background exposure

    Returns:
        master_bkgd (data.Background): Background frame
        fileprefix (str): if defined, a prefix to prepend to each saved file. If none, uses input filename.
        badpixmap (data.BadPixelMap): 2-D map of badpixels.
    """
    readnoisebar = False # temp ... TODO ask someone who knows more about detectors

    background_badpix_cube = []
    background_cube = []

    for frame in background_frames:

        background_badpixmap = get_badpixmap_from_laplacian(frame.data, bad_pixel_fraction=1e-2)
        background_badpixmap = background_badpixmap*get_badpixmap_from_mad(frame.data, threshold=10)
        if readnoisebar:
            background_badpixmap = background_badpixmap*get_badpixmap_from_readnoisebars(frame.data, frame.header)

        background_cube.append(frame.data*background_badpixmap)
        background_badpix_cube.append(background_badpixmap)

        if plot:
            plt.figure(2)
            plt.imshow(frame.data * background_badpixmap,interpolation="nearest",origin="lower")
            med_val = np.nanmedian(frame.data)
            plt.clim([0,2*med_val])
            plt.show()

    background_cube = np.array(background_cube)
    background_badpix_cube = np.array(background_badpix_cube)

    # suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bkgd_noise = np.nanstd(background_cube, axis=0)
        master_bkgd = np.nanmean(background_cube, axis=0)

    badpixmap = np.ones(master_bkgd.shape)
    badpixmap[np.where(np.nansum(background_badpix_cube,axis=0)<np.max([2,0.25*background_cube.shape[0]]))] = np.nan

    # make smoothed version of the master background noise
    smoothed_thermal_noise = ndi.median_filter(bkgd_noise, 3)
    smoothed_thermal_noise_percolumn = np.nanmedian(smoothed_thermal_noise, axis=0)
    smoothed_thermal_noise_percolumn_2d = np.ones(bkgd_noise.shape) * smoothed_thermal_noise_percolumn[None, :]
    bad_thermal = np.where(np.isnan(smoothed_thermal_noise*badpixmap))
    smoothed_thermal_noise[bad_thermal] = smoothed_thermal_noise_percolumn_2d[bad_thermal]

    bkgd_header = background_frames[0].header
    if fileprefix is None:
        # use original filename as prefix if no prefix given
        orig_filename = background_frames[0].filename[:-5] # strip off the .FITS
        fileprefix = orig_filename

    tint = float(bkgd_header["TRUITIME"])
    coadds = int(bkgd_header["COADDS"])
    if readnoisebar:
        bkgd_filename = fileprefix + "_background_med_tint{0}_coadds{1}.fits".format(tint,coadds)
        bpmap_filename = fileprefix + "_persistent_badpix_tint{0}_coadds{1}.fits".format(tint,coadds)
    else:
        bkgd_filename = fileprefix + "_background_med_nobars_tint{0}_coadds{1}.fits".format(tint,coadds)
        bpmap_filename = fileprefix + "_persistent_badpix_nobars_tint{0}_coadds{1}.fits".format(tint,coadds)

    bkgd_filepath = os.path.join(background_frames[0].filedir, bkgd_filename)
    bpmap_filepath = os.path.join(background_frames[0].filedir, bpmap_filename)

    master_bkgd = data.Background(data=master_bkgd, header=bkgd_header, filepath=bkgd_filepath, data_noise=smoothed_thermal_noise)
    badpixmap = data.BadPixelMap(data=badpixmap, header=bkgd_header, filepath=bpmap_filepath)

    # add history of data reduction
    tnow = time.Time.now()
    master_bkgd.header['HISTORY'] = "[{0}] Combined into background file".format(str(tnow))
    badpixmap.header['HISTORY'] = "[{0}] Combined into bad pixel file".format(str(tnow))
    # write to header all the files that were used in making this file
    for calib_frame in [master_bkgd, badpixmap]:
        calib_frame.header['DRPNFILE'] = len(background_frames)
        for i in range(len(background_frames)):
            calib_frame.header['FILE_{0}'.format(i)] = background_frames[0].filename

    return master_bkgd, badpixmap

def process_backgrounds(frames, plot=False, save_loc=None, fileprefix=None, caldb_save_loc=None):
    """
    Function to bach process a series of thermal background images taken at different exposure times
    Saves background and bad pixel maps to the directory specified by save_loc

    Args:
        frames (data.Dataset): Dataset of frames to be processed as background frames
        plot (bool): if True, plots the images. Default is false
        save_loc (str): if defined, the filepath to the directory to save the images.
        fileprefix (str): if defined, a prefix to prepend to each saved file. If none, uses input filename.  
        caldb_save_loc(DetectorCalDB object): if defined, the calibration database to keep track of calibrated files
    """

    # check through all the file headers to figure out what exposure times we have
    tint_list = []
    coadds_list = []
    for frame in frames:
        logging.info(frame.filename)
        background_header = frame.header
        tint_list.append(float(background_header["TRUITIME"]))
        coadds_list.append(int(background_header["COADDS"]))

    unique_tint = np.unique(tint_list)
    unique_coadds = np.unique(coadds_list)

    tint_outlist = []
    coadd_outlist = []

    background_meds = []
    persistent_badpixs = []
    for tint in unique_tint:
        for coadds in unique_coadds:
            logging.info("tint={0}".format(tint))
            logging.info("coadds={0}".format(coadds))
            where_tint = np.where((tint_list==tint)*(coadds_list==coadds))
            if np.size(where_tint[0]) == 0:
                continue

            tint_outlist.append(tint)
            coadd_outlist.append(coadds)

            logging.info("N files = {0}".format(np.size(where_tint[0])))
            
            background_files = frames[where_tint[0]]
            background_med, persistent_badpix = create_background_badpixelmap(background_files, fileprefix=fileprefix)

            background_meds.append(background_med)
            persistent_badpixs.append(persistent_badpix)

            if plot:
                plt.figure()
                plt.imshow(background_med.data, interpolation="nearest",origin="lower")
                plt.show()
            
            if save_loc is not None:
                background_med.save(filedir=save_loc)
                persistent_badpix.save(filedir=save_loc)
            
            if caldb_save_loc is not None:
                background_med.save(caldb=caldb_save_loc)
                persistent_badpix.save(caldb=caldb_save_loc)
            
            
    return(background_meds,persistent_badpixs,tint_outlist,coadd_outlist)

