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
from scipy.optimize import nnls
from scipy.optimize import lsq_linear
import csv

def get_err_from_posterior(x,posterior):
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior/np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    if len(x[0:argmax_post]) < 2:
        lx = np.nan
    else:
        lf = interp1d(cum_posterior[0:argmax_post],x[0:argmax_post],bounds_error=False,fill_value=np.nan)
        lx = lf(1-0.6827)
    if len(x[argmax_post::]) < 2:
        rx = np.nan
    else:
        rf = interp1d(cum_posterior[argmax_post::],x[argmax_post::],bounds_error=False,fill_value=np.nan)
        rx = rf(1-0.6827)
    return x[argmax_post],(rx-lx)/2.,argmax_post

if __name__ == "__main__":
    try:
        import mkl

        mkl.set_num_threads(1)
    except:
        pass


    mykpicdir = "/scr3/jruffio/data/kpic/"
    phoenix_folder = "/scr3/jruffio/data/kpic/models/phoenix/"
    fontsize = 15
    selec_orders = [5,6,7,8]
    sciencedir_list = []
    sciencedir_list.append(os.path.join(mykpicdir, "20200701_HR_8799_c"))
    sciencedir_list.append(os.path.join(mykpicdir, "20200702_HR_8799_d"))
    sciencedir_list.append(os.path.join(mykpicdir, "20200703_HR_8799_e"))
    names = ["c","d","e"]
    colors=["#ff9900", "#6600ff","#ff99cc"] # b: "#0099cc"
    linestyles = ["--",":","-"]
    for k,(sciencedir,name,color,ls) in enumerate(zip(sciencedir_list,names,colors,linestyles)):
        order_suffix = ""
        for myorder in selec_orders:
            order_suffix += "_{0}".format(myorder)
        out = os.path.join(sciencedir, "out", "flux_and_posterior"+order_suffix+".fits")
        with pyfits.open(out) as hdulist:
            fluxout = hdulist[0].data
            dAICout = hdulist[1].data
            logpostout = hdulist[2].data
            vsini_list = hdulist[3].data
            rv_list = hdulist[4].data

        print("scaling factor",np.min(fluxout[2,0,:,:]),np.max(fluxout[2,0,:,:]))
        argmaxvsini, argmaxrv = np.unravel_index(np.argmax(logpostout[0, :, :]), logpostout[0, :, :].shape)
        argmaxvsini = 0
        print(np.max(logpostout[0,:,:]),logpostout[0,argmaxvsini, argmaxrv])
        _fluxout = fluxout[0, :, argmaxvsini, :]  # /np.nanstd(fluxout[3::,argmaxvsini,:])
        _fluxout_err = fluxout[1, :, argmaxvsini, :]  # /np.nanstd(fluxout[3::,argmaxvsini,:])

        plt.figure(1, figsize=(12, 9))

        plt.subplot(2, 1, 1)
        tint_min = 10*len(glob(os.path.join(sciencedir,"raw","*.fits")))
        plt.plot(rv_list, _fluxout[0, :] / fluxout[1, 0, argmaxvsini, :], alpha=1, label="HR 8799 "+name + " (tot. exp. = {0}min)".format(tint_min),linestyle=ls, color=color,linewidth=3)
        if k==2:
            plt.plot(rv_list, _fluxout[2, :] / fluxout[1, 2, argmaxvsini, :], alpha=0.5, label= "Background",linestyle="--", color="black")
        else:
            plt.plot(rv_list, _fluxout[2, :] / fluxout[1, 2, argmaxvsini, :], alpha=0.5,linestyle="--", color="black")
        plt.plot(rv_list, _fluxout[3, :] / fluxout[1, 3, argmaxvsini, :], alpha=0.5,linestyle="--", color="black")
        plt.plot(rv_list, _fluxout[4, :] / fluxout[1, 4, argmaxvsini, :], alpha=0.5,linestyle="--", color="black")
        plt.ylabel("S/N",fontsize=fontsize)
        plt.xlabel("RV (km/s)",fontsize=fontsize)
        plt.tick_params(axis="x",labelsize=fontsize)
        plt.tick_params(axis="y",labelsize=fontsize)
        # plt.ylim([-3,5])
        # plt.subplot(2, 3, 3)
        # plt.imshow(fluxout[2,0,:,:],interpolation="nearest",origin="lower",extent=[rv_list[0],rv_list[-1],vsini_list[0],vsini_list[-1]])
        # plt.xlabel("RV (km/s)")
        # plt.xlabel("vsin(i) (km/s)")
        # plt.colorbar()

        post = np.exp(logpostout[0, :, :] - np.nanmax(logpostout[0, :, :]))
        dvsini_list = vsini_list[1::]-vsini_list[0:np.size(vsini_list)-1]
        dvsini_list = np.insert(dvsini_list,0,[dvsini_list[0]])
        drv_list = rv_list[1::]-rv_list[0:np.size(rv_list)-1]
        drv_list = np.insert(drv_list,0,[drv_list[0]])
        print(np.size(dvsini_list),np.size(vsini_list))
        print(np.size(drv_list),np.size(rv_list))

        plt.subplot(2, 1, 2)
        rvpost = np.nansum(post*dvsini_list[:,None], axis=0)
        bestrv,rverr,_=get_err_from_posterior(rv_list, rvpost)
        # plt.gca().text(rv_d[2] + 0.25, 1, "${0:.1f}\pm {1:.1f}$ km/s".format(rv_d[2], rverr_d[2]), ha="center",va="bottom", rotation=0, size=fontsize, color="#330099")
        plt.plot(rv_list, rvpost / np.nanmax(rvpost), label=name + ": ${0:.1f}\pm {1:.1f}$ km/s".format(bestrv, rverr),linestyle=ls, color=color,linewidth=3)
        plt.xlabel("RV (km/s)",fontsize=fontsize)
        plt.xlim([-20,0])
        plt.ylim([0,1.1])
        plt.ylabel("$\propto \mathcal{P}(RV|d)$",fontsize=fontsize)
        plt.tick_params(axis="x",labelsize=fontsize)
        plt.tick_params(axis="y",labelsize=fontsize)
        # plt.gca().spines["right"].set_visible(False)
        # plt.gca().spines["top"].set_visible(False)
        # plt.gca().spines["left"].set_position(("data",-20))
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize)#

        # plt.subplot(1, 3, 3)
        # vsinipost = np.nansum(post*drv_list[None,:], axis=1)
        # plt.plot(vsini_list, vsinipost / np.nanmax(vsinipost), label=name,linestyle="-", color=color)
        # # vsinicdf = np.cumsum(vsinipost*dvsini_list)
        # # plt.plot(vsini_list, vsinicdf / np.nanmax(vsinicdf),label="CDF")
        # # plt.legend()
        # plt.xlabel("vsin(i) (km/s)")
        # # plt.subplot(2, 3, 6)
        # # plt.imshow(logpostout[0,:,:],interpolation="nearest",origin="lower",extent=[rv_list[0],rv_list[-1],vsini_list[0],vsini_list[-1]])
        # # plt.xlabel("RV (km/s)")
        # # plt.xlabel("vsin(i) (km/s)")
        # # plt.colorbar()

    plt.subplot(2, 1,1)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    if 1:
        out = os.path.join("/scr3/jruffio/data/kpic/figures/","summary.png")
        print("Saving " + out)
        plt.savefig(out)
        plt.savefig(out.replace(".png",".pdf"))


    plt.show()
