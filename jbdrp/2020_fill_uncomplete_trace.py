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
import itertools
from utils_2020.badpix import *
from utils_2020.misc import *
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize


if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    Nfib = 4
    usershift=0
    fitbackground = False
    mykpicdir = "/scr3/jruffio/data/kpic/"
    # mydir = os.path.join(mykpicdir,"20200607_ups_Her")
    # mydir = os.path.join(mykpicdir,"20200608_zet_Aql")
    # mydir = os.path.join(mykpicdir,"20200608_d_Sco")
    # mydir = os.path.join(mykpicdir,"20200609_ups_Her")
    # mydir = os.path.join(mykpicdir,"20200609_d_Sco")
    # mydir = os.path.join(mykpicdir,"20200609_kap_And")
    # mydir = os.path.join(mykpicdir,"20200609_zet_Aql")
    mydir = os.path.join(mykpicdir,"20200607_5_Vul")
    mydate = os.path.basename(mydir).split("_")[0]

    # mydir_ref = os.path.join(mykpicdir,"20200609_ups_Her")
    mydir_ref = os.path.join(mykpicdir,"20200607_ups_Her")
    mydate_ref = os.path.basename(mydir_ref).split("_")[0]


    out = os.path.join(mydir, "calib", mydate + "_line_width_smooth.fits")
    hdulist = pyfits.open(out)
    incomp_smooth_width_calib = hdulist[0].data
    incomp_smooth_deltawidth_calib = hdulist[1].data
    incomp_header = hdulist[0].header
    out = os.path.join(mydir, "calib", mydate + "_trace_loc_smooth.fits")
    hdulist = pyfits.open(out)
    incomp_smooth_loc_calib = hdulist[0].data
    incomp_smooth_deltaloc_calib = hdulist[1].data
    incomp_header = hdulist[0].header

    print(incomp_smooth_width_calib.shape)
    missing_fibs = np.where(np.nansum(incomp_smooth_width_calib,axis=(1,2))==0)[0]
    print(missing_fibs)
    # exit()

    out = os.path.join(mydir_ref, "calib", mydate_ref + "_line_width_smooth.fits")
    hdulist = pyfits.open(out)
    ref_smooth_width_calib = hdulist[0].data
    ref_smooth_deltawidth_calib = hdulist[1].data
    ref_header = hdulist[0].header
    out = os.path.join(mydir_ref, "calib", mydate_ref + "_trace_loc_smooth.fits")
    hdulist = pyfits.open(out)
    ref_smooth_loc_calib = hdulist[0].data
    ref_smooth_deltaloc_calib = hdulist[1].data
    ref_header = hdulist[0].header

    for fib in missing_fibs:
        print(fib)
        incomp_smooth_width_calib[fib,:,:] = ref_smooth_width_calib[fib,:,:]
        incomp_smooth_deltawidth_calib[fib,:,:] = ref_smooth_deltawidth_calib[fib,:,:]
        incomp_smooth_loc_calib[fib,:,:] = ref_smooth_loc_calib[fib,:,:]
        incomp_smooth_deltaloc_calib[fib,:,:] = ref_smooth_deltaloc_calib[fib,:,:]

    if 1:
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=incomp_smooth_width_calib,header=incomp_header))
        hdulist.append(pyfits.ImageHDU(data=incomp_smooth_deltawidth_calib))
        out = os.path.join(mydir, "calib", os.path.basename(mydir)+"_filledwith_"+os.path.basename(mydir_ref)+"_line_width_smooth.fits")
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=incomp_smooth_loc_calib,header=incomp_header))
        hdulist.append(pyfits.ImageHDU(data=incomp_smooth_deltaloc_calib))
        out = os.path.join(mydir, "calib",  os.path.basename(mydir)+"_filledwith_"+os.path.basename(mydir_ref)+"_trace_loc_smooth.fits")
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        if 1: #plot
            trace_loc_filename = glob(os.path.join(mydir, "calib", os.path.basename(mydir)+"_filledwith_"+os.path.basename(mydir_ref)+"_trace_loc_smooth.fits"))[0]
            hdulist = pyfits.open(trace_loc_filename)
            trace_loc = hdulist[0].data
            trace_loc[np.where(trace_loc == 0)] = np.nan
            print(trace_loc.shape)
            # plt.figure(1)
            # for order_id in range(9):
            #     plt.subplot(9, 1, 9-order_id)
            #     plt.plot(trace_loc[1,order_id,:],linestyle="-",linewidth=2)
            #     plt.legend()
            # plt.show()

            trace_loc_slit = np.zeros((trace_loc.shape[0], trace_loc.shape[1], trace_loc.shape[2]))
            trace_loc_dark = np.zeros((trace_loc.shape[0] * 2, trace_loc.shape[1], trace_loc.shape[2]))
            for order_id in range(9):
                dy1 = np.nanmean(trace_loc[0, order_id, :] - trace_loc[1, order_id, :]) / 2
                dy2 = np.nanmean(trace_loc[0, order_id, :] - trace_loc[3, order_id, :])
                # exit()
                if np.isnan(dy1):
                    dy1 = 10
                if np.isnan(dy2):
                    dy2 = 40
                print(dy1, dy2)

                trace_loc_slit[0, order_id, :] = trace_loc[0, order_id, :] - dy1
                trace_loc_slit[1, order_id, :] = trace_loc[1, order_id, :] - dy1
                trace_loc_slit[2, order_id, :] = trace_loc[2, order_id, :] - dy1
                trace_loc_slit[3, order_id, :] = trace_loc[3, order_id, :] - dy1

                if order_id == 0:
                    trace_loc_dark[0, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 0 * dy1
                    trace_loc_dark[1, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 1 * dy1
                    trace_loc_dark[2, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 2 * dy1
                    trace_loc_dark[3, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 3 * dy1

                    trace_loc_dark[4, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 4 * dy1
                    trace_loc_dark[5, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 5 * dy1
                    trace_loc_dark[6, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 6 * dy1
                    trace_loc_dark[7, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 7 * dy1
                else:
                    trace_loc_dark[0, order_id, :] = trace_loc[0, order_id, :] - 3 * dy2 + 3 * dy1
                    trace_loc_dark[1, order_id, :] = trace_loc[1, order_id, :] - 3 * dy2 + 4 * dy1
                    trace_loc_dark[2, order_id, :] = trace_loc[2, order_id, :] - 3 * dy2 + 5 * dy1
                    trace_loc_dark[3, order_id, :] = trace_loc[3, order_id, :] - 3 * dy2 + 6 * dy1

                    trace_loc_dark[4, order_id, :] = trace_loc[0, order_id, :] - 2 * dy2 + 2 * dy1
                    trace_loc_dark[5, order_id, :] = trace_loc[1, order_id, :] - 2 * dy2 + 3 * dy1
                    trace_loc_dark[6, order_id, :] = trace_loc[2, order_id, :] - 2 * dy2 + 4 * dy1
                    trace_loc_dark[7, order_id, :] = trace_loc[3, order_id, :] - 2 * dy2 + 5 * dy1

            plt.figure(1)
            for order_id in range(9):
                for fib in range(trace_loc.shape[0]):
                    plt.plot(trace_loc[fib, order_id, :], label="fibers", color="cyan",linestyle="--",linewidth=1)
                plt.plot(trace_loc[0, order_id, :], label="fibers", color="cyan",linestyle="-",linewidth=2)
                for fib in np.arange(0,trace_loc_slit.shape[0]):
                    plt.plot(trace_loc_slit[fib, order_id, :], label="background", color="red",linestyle="-.",linewidth=1)
                plt.plot(trace_loc_slit[0, order_id, :], label="background", color="red",linestyle="-",linewidth=1)
                for fib in np.arange(0,trace_loc_dark.shape[0]):
                    plt.plot(trace_loc_dark[fib, order_id, :], label="dark", color="white",linestyle=":",linewidth=2)
                plt.plot(trace_loc_dark[0, order_id, :], label="dark", color="white",linestyle="-",linewidth=2)
            plt.show()
    exit()