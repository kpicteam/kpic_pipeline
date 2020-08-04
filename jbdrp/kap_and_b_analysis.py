from glob import glob
import os
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from utils.spectra import *

if __name__ == "__main__":
    try:
        import mkl

        mkl.set_num_threads(1)
    except:
        pass
    fontsize=12
    mykpicdir = "/scr3/jruffio/data/kpic/"
    out_pngs =  "/scr3/jruffio/data/kpic/figures/"
    if 1:
        target = "kap_And_B"
        rv_star,rverr_star  = -12.7,0.8
        vsini_star , vsinierr_star = np.nan,np.nan
        sciencedir = os.path.join(mykpicdir,"20191107_kap_And_B")
        filelist = glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        myfib = 2

    for filename in filelist:
        print(filename)
        hdulist = pyfits.open(filename)
        try:
            fluxout += hdulist[0].data
        except:
            fluxout = hdulist[0].data
        try:
            dAIC += hdulist[1].data
        except:
            dAIC = hdulist[1].data
        # dAIC = hdulist[1].data
        try:
            logpostout += hdulist[2].data
        except:
            logpostout = hdulist[2].data
        print(logpostout.shape)
        vsini_list = hdulist[3].data
        rv_list = hdulist[4].data
        print(vsini_list.shape,rv_list.shape)

        post = np.exp(logpostout[0,:,:] - np.nanmax(logpostout[0,:,:]))
        vsinipost = np.nansum(post, axis=1)
        rvpost = np.nansum(post, axis=0)

    plt.figure(1)
    plt.plot(rv_list,dAIC[0,0,:],label=os.path.basename(filename).replace("_fluxes_flux_and_posterior.fits",""))
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.plot(rv_list, rvpost / np.nanmax(rvpost),label=os.path.basename(filename).replace("_fluxes_flux_and_posterior.fits",""))
    plt.subplot(1, 2, 2)
    plt.plot(vsini_list, vsinipost / np.nanmax(vsinipost),label=os.path.basename(filename).replace("_fluxes_flux_and_posterior.fits",""))
    plt.show()

    exit()