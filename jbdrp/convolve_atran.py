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
from utils.spectra import *
from utils.misc import *


if __name__ == "__main__":
    try:
        import mkl

        mkl.set_num_threads(1)
    except:
        pass

    numthreads = 30
    mypool = mp.Pool(processes=numthreads)

    mykpicdir = "/scr3/jruffio/data/kpic/"
    mydir = os.path.join(mykpicdir, "20191107_kap_And")

    hdulist = pyfits.open(os.path.join(mydir, "calib", "special_calib", "wvs_f1.fits"))
    wvs_f0 = hdulist[0].data
    hdulist = pyfits.open(os.path.join(mydir, "calib", "special_calib", "wvs_f2.fits"))
    wvs_f1 = hdulist[0].data
    hdulist = pyfits.open(os.path.join(mydir, "calib", "special_calib", "wvs_f3.fits"))
    wvs_f2 = hdulist[0].data
    wvs = np.array([wvs_f0, wvs_f1, wvs_f2])
    print(wvs.shape)

    line_width_func_list = []
    pixel_width_func_list = []
    line_width_filename = glob(os.path.join(mydir, "calib", "*_line_width_smooth.fits"))[0]
    hdulist = pyfits.open(line_width_filename)
    line_width = hdulist[0].data
    for fib in range(3):
        dwvs = wvs[fib][:, 1:2048] - wvs[fib][:, 0:2047]
        dwvs = np.concatenate([dwvs, dwvs[:, -1][:, None]], axis=1)
        # for order_id in range(9):
        #     plt.subplot(9, 1, 9-order_id)
        #     plt.plot(line_width[fib,order_id,:],label="{0}".format(fib))
        line_width_wvunit = line_width[fib, :, :] * dwvs
        line_width_func = interp1d(np.ravel(wvs[fib]), np.ravel(line_width_wvunit), bounds_error=False,
                                   fill_value=np.nan)
        pixel_width_func = interp1d(np.ravel(wvs[fib]), np.ravel(dwvs), bounds_error=False, fill_value=np.nan)
        line_width_func_list.append(line_width_func)
        pixel_width_func_list.append(pixel_width_func)
    # plt.legend()
    # plt.show()

    for fib in range(3):
        atran_func_list = []
        # atran_filename = os.path.join("/scr3/jruffio/data/kpic/","models","atran","atran_13599_30_0_2_45_135_245_0.txt")
        atran_filelist = glob(os.path.join(mykpicdir, "models", "atran", "atran_13599_30_*.dat"))
        atran_filelist.sort()
        print(atran_filelist)
        water_list = np.array([int(atran_filename.split("_")[-5]) for atran_filename in atran_filelist])
        waterargsort = np.argsort(water_list).astype(np.int)
        water_list = np.array([water_list[k] for k in waterargsort])
        atran_filelist = [atran_filelist[k] for k in waterargsort]
        atran_spec_list = []
        for k, atran_filename in enumerate(atran_filelist):
            print(atran_filename)
            atran_arr = np.loadtxt(atran_filename).T
            atran_wvs = atran_arr[1, :]
            atran_spec = atran_arr[2, :]
            atran_line_widths = np.array(
                pd.DataFrame(line_width_func_list[fib](atran_wvs)).interpolate(method="linear").fillna(
                    method="bfill").fillna(method="ffill"))[:, 0]
            atran_pixel_widths = np.array(
                pd.DataFrame(pixel_width_func_list[fib](atran_wvs)).interpolate(method="linear").fillna(
                    method="bfill").fillna(method="ffill"))[:, 0]
            atran_spec_conv = convolve_spectrum_line_width(atran_wvs, atran_spec, atran_line_widths, mypool=mypool)
            atran_spec_conv2 = convolve_spectrum_pixel_width(atran_wvs, atran_spec_conv, atran_pixel_widths,
                                                             mypool=mypool)
            atran_spec_list.append(atran_spec_conv2)
            # plt.plot(atran_wvs,atran_spec,label=os.path.basename(atran_filename))
            # plt.plot(atran_wvs,atran_spec_conv,label=os.path.basename(atran_filename))
            # plt.plot(atran_wvs, atran_spec_conv2, label=os.path.basename(atran_filename))
            # plt.show()

        # print(np.array(atran_spec_list).shape)
        # print(water_list.shape,atran_wvs.shape)
        # atran_2d_func = interp2d(atran_wvs,water_list, np.array(atran_spec_list), bounds_error=False, fill_value=np.nan)
        # # print(atran_2d_func(wvs_per_orders[3,:],1.5*np.ones(wvs_per_orders[3,:].shape)).shape)
        # plt.plot(wvs_per_orders[3,:],atran_2d_func(wvs_per_orders[3,:],2000),label="interp")
        # plt.legend()

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=np.array(atran_spec_list)))
        out = os.path.join(mydir, "..", "models", "atran", "atran_spec_list_f{0}.fits".format(fib))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=np.array(water_list)))
        out = os.path.join(mydir, "..", "models", "atran", "water_list.fits")
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=np.array(atran_wvs)))
        out = os.path.join(mydir, "..", "models", "atran", "atran_wvs.fits")
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()


    mypool.close()
    mypool.join()
        # plt.show()
    exit()