import astropy.io.fits as pyfits
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import median_filter
from astropy.stats import mad_std
from scipy.signal import correlate2d
from copy import copy
from scipy.interpolate import interp1d
from scipy.ndimage.filters import convolve


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