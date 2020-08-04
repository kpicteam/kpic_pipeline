from astropy.io import fits
import numpy as np
import scipy.ndimage as ndi
from astropy.stats import mad_std
from scipy.interpolate import interp1d
from scipy.ndimage.filters import convolve

import logging
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

def make_badpixmap(background_files,plot=False):
    readnoisebar = False # temp ... TODO ask someone who knows more about detectors

    background_badpix_cube = []
    background_cube = []

    for filename in background_files:
        hdulist = fits.open(filename)
        background = hdulist[0].data.T[:,::-1]
        background_header = hdulist[0].header

        background_badpixmap = get_badpixmap_from_laplacian(background,bad_pixel_fraction=1e-2)
        background_badpixmap = background_badpixmap*get_badpixmap_from_mad(background,threshold=10)
        if readnoisebar:
            background_badpixmap = background_badpixmap*get_badpixmap_from_readnoisebars(background,background_header)

        background_cube.append(background*background_badpixmap)
        background_badpix_cube.append(background_badpixmap)

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(2)
            plt.imshow(background*background_badpixmap,interpolation="nearest",origin="lower")
            med_val = np.nanmedian(background)
            plt.clim([0,2*med_val])
            plt.show()

    background_cube = np.array(background_cube)
    background_badpix_cube = np.array(background_badpix_cube)

    bkgd_noise = np.nanstd(background_cube, axis=0)
    master_bkgd = np.nanmean(background_cube, axis=0)

    badpixmap = np.ones(master_bkgd.shape)
    badpixmap[np.where(np.nansum(background_badpix_cube,axis=0)<np.max([2,0.25*background_cube.shape[0]]))] = np.nan

    smoothed_thermal_noise = ndi.median_filter(bkgd_noise, 3)
    smoothed_thermal_noise_percolumn = np.nanmedian(smoothed_thermal_noise, axis=0)
    smoothed_thermal_noise_percolumn_2d = np.ones(bkgd_noise.shape) * smoothed_thermal_noise_percolumn[None, :]
    bad_thermal = np.where(np.isnan(smoothed_thermal_noise*badpixmap))
    smoothed_thermal_noise[bad_thermal] = smoothed_thermal_noise_percolumn_2d[bad_thermal]

    return master_bkgd, smoothed_thermal_noise, badpixmap

def process_backgrounds(filelist,plot=False,save_loc=None):
    tint_list = []
    coadds_list = []
    header_list = []
    for filename in filelist:
        logging.info(filename)
        background_header = fits.getheader(filename)
        tint_list.append(float(background_header["TRUITIME"]))
        coadds_list.append(int(background_header["COADDS"]))
        header_list.append(background_header)

    unique_tint = np.unique(tint_list)
    unique_coadds = np.unique(coadds_list)
    
    background_meds = []
    persistent_badpixs = []
    smoothed_thermal_noises = []
    for tint in unique_tint:
        for coadds in unique_coadds:
            logging.info("tint={0}".format(tint))
            logging.info("coadds={0}".format(coadds))
            where_tint = np.where((tint_list==tint)*(coadds_list==coadds))
            if np.size(where_tint[0]) == 0:
                continue
            logging.info("N files = {0}".format(np.size(where_tint[0])))
            
            background_files = np.array(filelist)[where_tint[0]]
            background_med, smoothed_thermal_noise, persistent_badpix = make_badpixmap(background_files)

            background_meds.append(background_med)
            persistent_badpixs.append(persistent_badpix)
            smoothed_thermal_noises.append(smoothed_thermal_noise)

            if plot:
                import matplotlib.pyplot as plt
                plt.figure(2)
                plt.imshow(background_med,interpolation="nearest",origin="lower")
                med_val = np.nanmedian(background_med)
                # plt.clim([0,2*med_val])
                plt.show()

            if save_loc is not None:
                save_bkgd_badpix(background_med,persistent_badpix,smoothed_thermal_noise,header_list[where_tint[0][0]],readnoisebar=False)

                

    return(background_meds,persistent_badpixs,smoothed_thermal_noises,unique_tint,unique_coadds)

def save_bkgd_badpix(master_bkgd,badpixmap,smoothed_thermal_noise,header,readnoisebar=False):
    tint = float(header["TRUITIME"])
    coadds = int(header["COADDS"])

    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU(data=master_bkgd,header=header)
    hdulist.append(fits.ImageHDU(data=smoothed_thermal_noise))
    if readnoisebar:
        out = save_loc+"_background_med_tint{0}_coadds{1}.fits".format(tint,coadds)
    else:
        out = save_loc+"_background_med_nobars_tint{0}_coadds{1}.fits".format(tint,coadds)
    try:
        hdulist.writeto(out, overwrite=True)
    except TypeError:
        hdulist.writeto(out, clobber=True)
    hdulist.close()

    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU(data=badpixmap,header=header)
    if readnoisebar:
        out = save_loc+"_persistent_badpix_tint{0}_coadds{1}.fits".format(tint,coadds)
    else:
        out = save_loc+"_persistent_badpix_nobars_tint{0}_coadds{1}.fits".format(tint,coadds)
    try:
        hdulist.writeto(out, overwrite=True)
    except TypeError:
        hdulist.writeto(out, clobber=True)
    hdulist.close()

# # EXAMPLE:

# from glob import glob
# import os
# from astropy.io import fits
# from kpicdrp import background

# # For multiple tint and coadds in one folder
# mykpicdir = "../kpic/"
# bkgddir = os.path.join(mykpicdir,"20200702_backgrounds")
# filelist = glob(os.path.join(mydir,"raw","*.fits"))

# # For multiple tint and coadds in one folder
# master_bkgds,badpixmaps,smoothed_thermal_noises,unique_tint,unique_coadds = backgrounds.process_backgrounds(filelist,plot=False,save_loc=bkgddir)

# # OR

# # For a single tint/number of coadds
# master_bkgd, smoothed_thermal_noise, badpixmap = backgrounds.make_badpixmap(filelist,plot=False) # does not save automatically
# save_bkgd_badpix(master_bkgd,badpixmap,smoothed_thermal_noise,header=fits.getheader(filelist[0]),readnoisebar=False)

