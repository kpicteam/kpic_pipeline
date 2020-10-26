import astropy.io.fits as pyfits
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import median_filter
from astropy.stats import mad_std
from scipy.signal import correlate2d
from copy import copy
import multiprocessing as mp
from scipy.optimize import minimize
import itertools
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import InterpolatedUnivariateSpline
from jbdrp.utils_2020.spectra import *
from jbdrp.utils_2020.misc import *
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from scipy import interpolate

if __name__ == "__main__":
    try:
        import mkl

        mkl.set_num_threads(1)
    except:
        pass

    numthreads = 30
    mypool = mp.Pool(processes=numthreads)

    mykpicdir = "/scr3/jruffio/data/kpic/"
    mydir = os.path.join(mykpicdir,"20191107_kap_And")
    target_rv = -12.70 # kap And

    mydate = os.path.basename(mydir).split("_")[0]

    hdulist = pyfits.open(os.path.join(mykpicdir,"special_calib","wvs_f1.fits"))
    wvs_f0 = hdulist[0].data
    hdulist = pyfits.open(os.path.join(mykpicdir,"special_calib","wvs_f2.fits"))
    wvs_f1 = hdulist[0].data
    hdulist = pyfits.open(os.path.join(mykpicdir,"special_calib","wvs_f3.fits"))
    wvs_f2 = hdulist[0].data
    wvs = np.array([wvs_f0,wvs_f1,wvs_f2])
    print(wvs.shape)

    # plt.figure(2)
    # plt.plot(np.ravel(wvs_f1))
    # plt.show()

    filelist = glob(os.path.join(mydir,"*fluxes.fits"))

    fluxes_list = []
    errors_list = []
    fiber_list = []
    header_list = []
    baryrv_list = []
    for filename in filelist:#
        print(filename)
        hdulist = pyfits.open(filename)
        fluxes = hdulist[0].data
        header = hdulist[0].header
        try:
            baryrv_list.append(header["BARYRV"])
        except:
            continue
        errors = hdulist[1].data
        fiber_list.append(np.argmax(np.nansum(fluxes,axis=(1,2))))
        fluxes_list.append(fluxes)
        errors_list.append(errors)
        header_list.append(header)
    fiber_list = np.array(fiber_list)
    fluxes_list = np.array(fluxes_list)
    errors_list = np.array(errors_list)
    baryrv_list = np.array(baryrv_list)
    print(fluxes_list.shape)
    print(fiber_list)

    combined_spec = np.zeros((3,9,2048))
    combined_spec_sig = np.zeros((3,9,2048))
    baryrv = np.zeros(3)
    for fib in np.unique(fiber_list):
        spec_list = fluxes_list[np.where(fib == fiber_list)[0],fib,:,:]
        spec_sig_list = errors_list[np.where(fib == fiber_list)[0],fib,:,:]
        baryrv[fib] = np.mean(baryrv_list[np.where(fib == fiber_list)[0]])
        # plt.figure(1)
        # for file_id in range(spec_list.shape[0]):
        #     for order_id in range(9):
        #         plt.subplot(9, 1, 9-order_id)
        #         plt.plot(spec_list[file_id,order_id,:],linestyle="-",linewidth=2,label="data {0}".format(file_id))
        #         plt.plot(spec_sig_list[file_id,order_id,:],linestyle="-",linewidth=2,label="error {0}".format(file_id))
        #         plt.legend()
        # plt.show()
        combined_spec[fib,:,:], combined_spec_sig[fib,:,:] = combine_stellar_spectra(spec_list,spec_sig_list)

    combined_spec = edges2nans(combined_spec)
    combined_spec_sig = edges2nans(combined_spec_sig)

    line_width_func_list = []
    pixel_width_func_list = []
    line_width_filename = glob(os.path.join(mydir,"calib","*_line_width_smooth.fits"))[0]
    hdulist = pyfits.open(line_width_filename)
    line_width = hdulist[0].data
    for fib in range(3):
        dwvs = wvs[fib][:, 1:2048] - wvs[fib][:, 0:2047]
        dwvs = np.concatenate([dwvs, dwvs[:, -1][:, None]], axis=1)
        line_width_wvunit = line_width[fib,:,:] * dwvs
        line_width_func = interp1d(np.ravel(wvs[fib]), np.ravel(line_width_wvunit), bounds_error=False,fill_value=np.nan)
        pixel_width_func = interp1d(np.ravel(wvs[fib]), np.ravel(dwvs), bounds_error=False, fill_value=np.nan)
        line_width_func_list.append(line_width_func)
        pixel_width_func_list.append(pixel_width_func)


    phoenix_folder = "/scr3/jruffio/data/kpic/models/phoenix/"
    phoenix_wv_filename = os.path.join(phoenix_folder, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
    with pyfits.open(phoenix_wv_filename) as hdulist:
        phoenix_wvs = hdulist[0].data / 1.e4
    crop_phoenix = np.where((phoenix_wvs > 1.8 - (2.6 - 1.8) / 2) * (phoenix_wvs < 2.6 + (2.6 - 1.8) / 2))
    phoenix_wvs = phoenix_wvs[crop_phoenix]
    phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "kap_And" + "*.fits"))[0]
    with pyfits.open(phoenix_model_host_filename) as hdulist:
        phoenix_A0 = hdulist[0].data[crop_phoenix]
    phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "lte05300" + "*.fits"))[0]
    with pyfits.open(phoenix_model_host_filename) as hdulist:
        phoenix_G7 = hdulist[0].data[crop_phoenix]
    phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "bet_Peg" + "*.fits"))[0]
    with pyfits.open(phoenix_model_host_filename) as hdulist:
        phoenix_M0 = hdulist[0].data[crop_phoenix]
    phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "DH_Tau" + "*.fits"))[0]
    with pyfits.open(phoenix_model_host_filename) as hdulist:
        phoenix_tetUMi = hdulist[0].data[crop_phoenix]



    phoenix_line_widths = np.array(
        pd.DataFrame(line_width_func_list[fib](phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(
            method="ffill"))[:, 0]
    phoenix_pixel_widths = np.array(
        pd.DataFrame(pixel_width_func_list[fib](phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(
            method="ffill"))[:, 0]
    phoenix_A0_conv = convolve_spectrum_line_width(phoenix_wvs, phoenix_A0, phoenix_line_widths, mypool=mypool)
    phoenix_A0_conv = convolve_spectrum_pixel_width(phoenix_wvs, phoenix_A0_conv, phoenix_pixel_widths, mypool=mypool)
    phoenix_A0_func = interp1d(phoenix_wvs, phoenix_A0_conv / np.nanmax(phoenix_A0_conv), bounds_error=False,
                               fill_value=np.nan)
    phoenix_G7_conv = convolve_spectrum_line_width(phoenix_wvs, phoenix_G7, phoenix_line_widths, mypool=mypool)
    phoenix_G7_conv = convolve_spectrum_pixel_width(phoenix_wvs, phoenix_G7_conv, phoenix_pixel_widths, mypool=mypool)
    phoenix_G7_func = interp1d(phoenix_wvs, phoenix_G7_conv / np.nanmax(phoenix_G7_conv), bounds_error=False,
                               fill_value=np.nan)
    phoenix_M0_conv = convolve_spectrum_line_width(phoenix_wvs, phoenix_M0, phoenix_line_widths, mypool=mypool)
    phoenix_M0_conv = convolve_spectrum_pixel_width(phoenix_wvs, phoenix_M0_conv, phoenix_pixel_widths, mypool=mypool)
    phoenix_M0_func = interp1d(phoenix_wvs, phoenix_M0_conv / np.nanmax(phoenix_M0_conv), bounds_error=False,
                               fill_value=np.nan)
    phoenix_tetUMi_conv = convolve_spectrum_line_width(phoenix_wvs, phoenix_tetUMi, phoenix_line_widths, mypool=mypool)
    phoenix_tetUMi_conv = convolve_spectrum_pixel_width(phoenix_wvs, phoenix_tetUMi_conv, phoenix_pixel_widths, mypool=mypool)
    phoenix_tetUMi_func = interp1d(phoenix_wvs, phoenix_tetUMi_conv / np.nanmax(phoenix_tetUMi_conv), bounds_error=False,
                               fill_value=np.nan)

    with open("/scr3/jruffio/data/kpic/models/planets_templates/lte018-5.0-0.0a+0.0.BT-Settl.spec.7", 'r') as f:
        model_wvs = []
        model_fluxes = []
        for line in f.readlines():
            line_args = line.strip().split()
            model_wvs.append(float(line_args[0]))
            model_fluxes.append(float(line_args[1].replace('D', 'E')))
    model_wvs = np.array(model_wvs) / 1.e4
    model_fluxes = np.array(model_fluxes)
    model_fluxes = 10 ** (model_fluxes - 8)
    crop_plmodel = np.where((model_wvs > 1.8 - (2.6 - 1.8) / 2) * (model_wvs < 2.6 + (2.6 - 1.8) / 2))
    model_wvs = model_wvs[crop_plmodel]
    model_fluxes = model_fluxes[crop_plmodel]
    # planet_convspec = convolve_spectrum(wmod, ori_planet_spec, 30000, specpool)
    if 1:
        pl_line_widths = np.array(
            pd.DataFrame(line_width_func(model_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(
                method="ffill"))[:, 0]
        pl_pixel_widths = np.array(
            pd.DataFrame(pixel_width_func(model_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(
                method="ffill"))[:, 0]
        planet_convspec = convolve_spectrum_line_width(model_wvs, model_fluxes, pl_line_widths, mypool=mypool)
        planet_convspec = convolve_spectrum_pixel_width(model_wvs, planet_convspec, pl_pixel_widths, mypool=mypool)
    planet_convspec /= np.nanmean(planet_convspec)
    phoenix_L_spline = interpolate.splrep(model_wvs, planet_convspec)



    out = os.path.join(mydir, "..", "models", "atran", "atran_spec_list_f{0}.fits".format(fib))
    hdulist = pyfits.open(out)
    atran_spec_list = hdulist[0].data
    out = os.path.join(mydir, "..", "models", "atran", "water_list.fits")
    hdulist = pyfits.open(out)
    water_list = hdulist[0].data
    out = os.path.join(mydir, "..", "models", "atran", "atran_wvs.fits")
    hdulist = pyfits.open(out)
    atran_wvs = hdulist[0].data
    atran_2d_func = interp2d(atran_wvs, water_list, np.array(atran_spec_list), bounds_error=False, fill_value=np.nan)

    print(water_list)
    plt.figure(1)
    spec = atran_2d_func(np.ravel(wvs_f1), water_list[2])
    plt.plot(np.ravel(wvs_f1),spec/np.max(spec[6200:12000]),label="low water telluric")
    spec = atran_2d_func(np.ravel(wvs_f1), water_list[3])
    plt.plot(np.ravel(wvs_f1),spec/np.max(spec[6200:12000]),label="high water telluric")
    # spec = phoenix_A0_func(np.ravel(wvs_f1))
    # plt.plot(np.ravel(wvs_f1),spec/np.max(spec[6200:12000]),label="Phoenix A0")
    # spec = phoenix_G7_func(np.ravel(wvs_f1))
    # plt.plot(np.ravel(wvs_f1),spec/np.max(spec[6200:12000]),label="Phoenix G7")
    spec = phoenix_M0_func(np.ravel(wvs_f1))
    plt.plot(np.ravel(wvs_f1),spec/np.max(spec[6200:12000]),label="Phoenix M0")
    spec = phoenix_tetUMi_func(np.ravel(wvs_f1))
    plt.plot(np.ravel(wvs_f1),spec/np.max(spec[6200:12000]),label="Phoenix tetUMi")
    # spec = interpolate.splev(np.ravel(wvs_f1), phoenix_L_spline, der=0)
    # plt.plot(np.ravel(wvs_f1),spec/np.max(spec[6200:12000]),label="Phoenix L")
    # spec = np.ravel(combined_spec[1,:,:])
    # plt.plot(np.ravel(wvs_f1),spec/np.nanmax(spec[6200:12000]),label="data")
    plt.legend()

    # plt.figure(2)
    # plt.plot(np.ravel(wvs_f1))
    plt.show()