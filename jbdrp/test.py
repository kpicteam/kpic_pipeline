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
from wavcal import convolve_spectrum_line_width,convolve_spectrum_pixel_width
from scipy.interpolate import interp1d
from utils_2020.badpix import *
from utils_2020.misc import *
from utils_2020.spectra import *
from scipy import interpolate
from PyAstronomy import pyasl
import csv

def LPFvsHPF(myvec,cutoff):
    myvec_cp = copy(myvec)
    #handling nans:
    wherenans = np.where(np.isnan(myvec_cp))
    window = int(round(np.size(myvec_cp)/(cutoff/2.)/2.))#cutoff
    tmp = np.array(pd.DataFrame(np.concatenate([myvec_cp, myvec_cp[::-1]], axis=0)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))
    myvec_cp_lpf = np.array(pd.DataFrame(tmp).rolling(window=window, center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[0:np.size(myvec), 0]
    myvec_cp[wherenans] = myvec_cp_lpf[wherenans]


    fftmyvec = np.fft.fft(np.concatenate([myvec_cp, myvec_cp[::-1]], axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec


    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan

    # plt.figure(10)
    # plt.plot(myvec_cp,label="fixed")
    # plt.plot(myvec,label="ori")
    # plt.plot(myvec_cp_lpf,label="lpf")
    # plt.plot(LPF_myvec,label="lpf fft")
    # plt.legend()
    # plt.show()
    return LPF_myvec,HPF_myvec

mykpicdir = "/scr3/jruffio/data/kpic/"
phoenix_folder = "/scr3/jruffio/data/kpic/models/phoenix/"

A0dir = os.path.join(mykpicdir, "20200702_d_Sco")
A0_rv = -13  # km/s
filelist = glob(os.path.join(A0dir, "*fluxes.fits"))
A0_spec,A0_err,_,_,A0_baryrv = combine_spectra_from_folder(filelist,"star")

# sciencedir = os.path.join(mykpicdir, "20200702_ROXs_42Bb")
sciencedir = os.path.join(mykpicdir, "20200702_ROXs_42B")
# sciencedir = os.path.join(mykpicdir,"20200702_ROXs_42B_daytime") #[ 15.93993899 305.19731602  15.53791296  14.55284422]
# sciencedir = os.path.join(mykpicdir,"20200702_ROXs_42B_fit") #[ 14.45913059 304.17032175  14.0525108   13.23600295]
# sciencedir = os.path.join(mykpicdir,"20200702_ROXs_42B_chopping") #[ 71.24556938 316.18770182 167.29706758  59.05578673]
filelist = glob(os.path.join(sciencedir, "*fluxes.fits"))
filelist.sort()
filelist = [filelist[0]]
print(filelist)
science_spec,science_err,slit_spec,dark_spec,science_baryrv = combine_spectra_from_folder(filelist,"science")

fib = 1
order = 4
cutoff = 40

# plt.plot(science_spec[fib,order,:]/np.nanmax(science_spec[fib,order,:]),label="science")
# plt.plot(A0_spec[fib,order,:]/np.nanmax(A0_spec[fib,order,:]),label="A0")

a = LPFvsHPF(science_spec[fib,order,:],cutoff)[1]
b = LPFvsHPF(A0_spec[fib,order,:],cutoff)[1]

plt.plot(a/np.nanmax(a),label="science")
plt.plot(b/np.nanmax(b),label="A0")

plt.legend()
plt.show()