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
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u


def instr_trans_model(paras, x,wvs0,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_model,rv):
    c_kms = 299792.458
    coefs = paras[0:deg_model+1]
    water = paras[deg_model+1]

    if 0:
        instr_trans= np.polyval(coefs,x)
    elif deg_model <=2:
        M2 = np.zeros((np.size(x),(deg_model+1)))
        x_knots = x[np.linspace(0,len(x)-1,deg_model+1,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
        for polypower in range(deg_model):
            if polypower == 0:
                where_chunk = np.where((x_knots[polypower]<=x)*(x<=x_knots[polypower+1]))
            else:
                where_chunk = np.where((x_knots[polypower]<x)*(x<=x_knots[polypower+1]))
            M2[where_chunk[0],polypower] = 1-(x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
            M2[where_chunk[0],1+polypower] = (x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
        instr_trans = np.dot(M2,coefs)
    else:
        x_knots = x[np.linspace(0,len(x)-1,deg_model+1,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
        spl = InterpolatedUnivariateSpline(x_knots,coefs,k=3,ext=0)
        instr_trans= spl(x)

    tmp = phoenix_HR8799_func(wvs0*(1-rv/c_kms))*atran_2d_func(wvs0, water)
    tmp /= np.nanmax(tmp)

    return tmp * instr_trans

def instr_trans_nloglike_poly(paras, x,wvs0,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_model,rv):
    m = instr_trans_model(paras, x,wvs0,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_model,rv)
    if m is not None:
        nloglike = np.nansum((spectrum - m) ** 2 / (spec_err) ** 2)
        return 1 / 2. * nloglike
    else:
        return np.inf

def _fit_instrument_transmission(paras):
    x,wvs0,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_model,rv = paras

    specmax = np.nanmax(spectrum)
    paras0 = np.array((specmax*np.ones(deg_model+1)).tolist() + [8000])
    simplex_init_steps =  np.ones(np.size(paras0))
    simplex_init_steps[0:deg_model+1] = specmax*0.1
    simplex_init_steps[deg_model+1] = 200
    initial_simplex = np.concatenate([paras0[None,:],paras0[None,:] + np.diag(simplex_init_steps)],axis=0)
    res = minimize(lambda paras: instr_trans_nloglike_poly(paras, x,wvs0,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_model,rv), paras0, method="nelder-mead",
                           options={"xatol": 1e-10, "maxiter": 1e5,"initial_simplex":initial_simplex,"disp":False})
    out = res.x
    print(res)
    # print(paras0)
    print(out)
    # plt.plot(x,wavcal_model_poly(paras0, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_model,deg_cont),label="model0")
    plt.fill_between(x,spectrum-spec_err,spectrum+spec_err,label="data err",alpha=0.5)
    plt.plot(x,spectrum,label="data",alpha=0.5)
    plt.plot(x,instr_trans_model(out, x,wvs0,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_model,rv),label="model")


    x_knots = x[np.linspace(0, len(x) - 1, deg_model + 1, endpoint=True).astype(np.int)]  # np.array([wvs_stamp[wvid] for wvid in )
    spl = InterpolatedUnivariateSpline(x_knots, out[0:deg_model+1], k=3, ext=0)

    plt.plot(x,spl(x) )
    plt.legend()
    plt.show()

    return spl(x),out,x_knots

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

    # plt.figure(1)
    # for fib in range(3):
    #     for order_id in range(9):
    #         plt.subplot(9, 1, 9-order_id)
    #         plt.plot(combined_spec[fib,order_id,:],linestyle="-",linewidth=2,label="data {0}".format(fib))
    #         plt.plot(combined_spec_sig[fib,order_id,:],linestyle="-",linewidth=2,label="error {0}".format(fib))
    #         plt.legend()
    # plt.show()


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


    transmission = np.zeros(line_width.shape)

    for fib in range(line_width.shape[0]):

        phoenix_line_widths = np.array(pd.DataFrame(line_width_func_list[fib](phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
        phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func_list[fib](phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
        phoenix_A0_conv = convolve_spectrum_line_width(phoenix_wvs, phoenix_A0, phoenix_line_widths, mypool=mypool)
        phoenix_A0_conv = convolve_spectrum_pixel_width(phoenix_wvs, phoenix_A0_conv, phoenix_pixel_widths,mypool=mypool)
        phoenix_A0_func = interp1d(phoenix_wvs, phoenix_A0_conv / np.nanmax(phoenix_A0_conv), bounds_error=False,fill_value=np.nan)

        out = os.path.join(mydir, "..", "models", "atran", "atran_spec_list_f{0}.fits".format(fib))
        hdulist = pyfits.open(out)
        atran_spec_list = hdulist[0].data
        out = os.path.join(mydir, "..", "models", "atran", "water_list.fits")
        hdulist = pyfits.open(out)
        water_list = hdulist[0].data
        out = os.path.join(mydir, "..", "models", "atran", "atran_wvs.fits")
        hdulist = pyfits.open(out)
        atran_wvs = hdulist[0].data
        atran_2d_func = interp2d(atran_wvs, water_list, np.array(atran_spec_list), bounds_error=False,fill_value=np.nan)

        if 1:
            l = 0
            x = np.arange(0,2048)
            spectrum = combined_spec[fib,l,:]
            spec_err = combined_spec_sig[fib,l,:]
            # spec_err = np.clip(spec_err,0.5*np.nanmedian(spec_err),np.inf)
            wvs0 = wvs[fib,l,:]
            deg_model = 5
            rv =  target_rv-baryrv[fib]

            new_wvs,out,x_knots = _fit_instrument_transmission((x,wvs0,spectrum,spec_err, phoenix_A0_func,atran_2d_func,deg_model,rv))
            print(out)
            exit()
        else:
            x = np.arange(0,2048)
            deg_model = 5
            rv =  target_rv-baryrv[fib]

            wvs0_chunks = []
            spectrum_chunks = []
            spec_err_chunks = []
            for l in range(9):
                wvs0_chunks.append(wvs[fib,l,:])
                spectrum_chunks.append(combined_spec[fib,l,:])
                spec_err = combined_spec_sig[fib,l,:]
                # spec_err = np.clip(spec_err,0.5*np.nanmedian(spec_err),np.inf)
                spec_err_chunks.append(spec_err)
            outputs_list = mypool.map(_fit_instrument_transmission, zip(itertools.repeat(x),
                                                          wvs0_chunks,spectrum_chunks,spec_err_chunks,
                                                          itertools.repeat(phoenix_A0_func),
                                                          itertools.repeat(atran_2d_func),
                                                          itertools.repeat(deg_model),
                                                          itertools.repeat(rv)))
            for l, out in enumerate(outputs_list):
                inst_trans,out,x_knots = out
                transmission[fib,l,:] = inst_trans

    print(transmission.shape)
    transmission/=np.nanmax(transmission,axis=(1,2))[:,None,None]

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=transmission,header=header_list[0]))
    out = os.path.join(mydir, "calib", mydate+"_instr_trans.fits")
    print(out)
    try:
        hdulist.writeto(out, overwrite=True)
    except TypeError:
        hdulist.writeto(out, clobber=True)
    hdulist.close()

    plt.figure(1)
    for fib in range(3):
        for l in range(transmission.shape[1]):
            plt.subplot(9, 1, 9 - l)
            plt.plot(transmission[fib,l, :])

    plt.show()
    exit()