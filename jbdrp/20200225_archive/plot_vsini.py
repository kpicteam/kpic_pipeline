import astropy.io.fits as pyfits
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import multiprocessing as mp
import pandas as pd
from scipy.interpolate import interp1d
import itertools
from PyAstronomy import pyasl
from scipy import interpolate


if __name__ == "__main__":
    data_dir = "/scr3/jruffio/data/kpic/HDC_workshop_data/"
    BD = "B" #"A" or "B" for 2M0746A or 2M0746B
    out_vsini_post_list = []
    for order in [0,1,2,6,7,8]:#range(9):
        selec_orders = [order]
        cutoff = 20
        c_kms = 299792.458
        rv_list = np.concatenate([np.arange(-300,-40,3),np.arange(-40,40,0.5),np.arange(40,300,3)], axis=0)
        # rv_list = np.arange(20,30,0.5)
        vsini_list = np.arange(20,50, 1)
        # Save data
        if 1:
            out = os.path.join(data_dir,"out", "2M0746"+BD+"_combined_rvs_order{0}.fits".format(order))
            hdulist = pyfits.open(out)
            rv_list = hdulist[0].data
            out = os.path.join(data_dir,"out", "2M0746"+BD+"_combined_vsini_order{0}.fits".format(order))
            hdulist = pyfits.open(out)
            vsini_list = hdulist[0].data
            out = os.path.join(data_dir,"out", "2M0746"+BD+"_combined_vsini_dAIC_order{0}.fits".format(order))
            hdulist = pyfits.open(out)
            out_dAIC = hdulist[0].data
            out = os.path.join(data_dir,"out", "2M0746"+BD+"_combined_vsini_ampl_order{0}.fits".format(order))
            hdulist = pyfits.open(out)
            out_ampl = hdulist[0].data
            out = os.path.join(data_dir,"out", "2M0746"+BD+"_combined_vsini_logpost_order{0}.fits".format(order))
            hdulist = pyfits.open(out)
            out_logpost = hdulist[0].data
            out_post = np.exp(out_logpost-np.max(out_logpost))
            out_vsini_post = np.sum(out_post,axis=0)
            out_RV_post = np.sum(out_post,axis=1)
            out_vsini_post_list.append(out_vsini_post)

        plt.figure(1)
        plt.plot(vsini_list,out_vsini_post/np.max(out_vsini_post),label="{0}".format(order))
        plt.legend()
        plt.figure(2)
        plt.plot(rv_list,out_RV_post/np.max(out_RV_post),label="{0}".format(order))
        plt.legend()

    plt.figure(1)
    plt.xlabel("vsini (km/s)")
    plt.ylabel("posterior")
    plt.savefig(os.path.join(data_dir, "out", "2M0746"+BD+"_combined_plot_vsini.png"), bbox_inches='tight')
    plt.figure(2)
    plt.xlabel("RV (km/s)")
    plt.ylabel("posterior")
    plt.xlim([-30,70])
    plt.savefig(os.path.join(data_dir, "out", "2M0746"+BD+"_combined_plot_RV.png"), bbox_inches='tight')
    plt.figure(3)
    plt.plot(np.prod(out_vsini_post_list,axis=0))
    plt.show()

        # plt.subplot(1,3,1)
        # plt.imshow(out_dAIC,extent=[vsini_list[0],vsini_list[-1],rv_list[0],rv_list[-1]],origin="lower")
        # plt.subplot(1,3,2)
        # plt.imshow(out_ampl,extent=[vsini_list[0],vsini_list[-1],rv_list[0],rv_list[-1]],origin="lower")
        # plt.subplot(1,3,3)
        # plt.imshow(np.exp(out_logpost-np.max(out_logpost)),extent=[vsini_list[0],vsini_list[-1],rv_list[0],rv_list[-1]],origin="lower")
        # plt.xlabel("vsini (km/s)")
        # plt.xlabel("RV (km/s)")
        # plt.show()
