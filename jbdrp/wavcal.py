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
from jbdrp.utils.badpix import *
from jbdrp.utils.misc import *
from jbdrp.utils.spectra import *

def wavcal_model_poly(paras, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitsrv,rv):
    c_kms = 299792.458
    wvs_coefs = paras[0:deg_wvs+1]
    sig = 1
    if fitsrv:
        star_rv = paras[deg_wvs+1]
        water = paras[deg_wvs+2]
    else:
        star_rv = rv
        water = paras[deg_wvs+1]

    if 0:
        wvs= np.polyval(wvs_coefs,x)
    elif deg_wvs <=2:
        M2 = np.zeros((np.size(x),(deg_wvs+1)))
        x_knots = x[np.linspace(0,len(x)-1,deg_wvs+1,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
        for polypower in range(deg_wvs):
            if polypower == 0:
                where_chunk = np.where((x_knots[polypower]<=x)*(x<=x_knots[polypower+1]))
            else:
                where_chunk = np.where((x_knots[polypower]<x)*(x<=x_knots[polypower+1]))
            M2[where_chunk[0],polypower] = 1-(x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
            M2[where_chunk[0],1+polypower] = (x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
        wvs = np.dot(M2,wvs_coefs)
    else:
        x_knots = x[np.linspace(0,len(x)-1,deg_wvs+1,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
        spl = InterpolatedUnivariateSpline(x_knots,wvs_coefs,k=3,ext=0)
        wvs= spl(x)
        # plt.plot(x_knots,wvs_coefs)
        # plt.plot(x,wvs)
        # plt.show()



    M = np.zeros((np.size(x),(deg_cont+1)))
    tmp = phoenix_HR8799_func(wvs*(1-star_rv/c_kms))*atran_2d_func(wvs, water)
    M0_mn = np.nanmean(tmp)
    tmp /= M0_mn
    # for stamp_id,(wvs_stamp,spec_stamp,first_index,last_index) in enumerate(zip(wvs_stamps,spec_stamps,first_index_list,last_index_list)):
    x_knots = x[np.linspace(0,len(x)-1,deg_cont+1,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
    # print(x_knots)
    # print(wvs_stamp)
    for polypower in range(deg_cont):
        if polypower == 0:
            where_chunk = np.where((x_knots[polypower]<=x)*(x<=x_knots[polypower+1]))
        else:
            where_chunk = np.where((x_knots[polypower]<x)*(x<=x_knots[polypower+1]))
        M[where_chunk[0],polypower] = 1-(x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
        M[where_chunk[0],1+polypower] = (x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
            # print("coucou")
    M = tmp[:,None]*M*instr_trans[:,None]
    # print(M.shape)
    # plt.plot(x,m,label="model")
    # plt.plot(x,spectrum,label="data")
    # plt.fill_between(x,spectrum-spec_err,spectrum+spec_err,label="data err",alpha=0.5)
    # plt.legend()
    # plt.show()

    if 1:
        deg_off = 1
        Moff = np.zeros((np.size(x),(deg_off+1)))
        x_knots = x[np.linspace(0,len(x)-1,deg_off+1,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
        for polypower in range(deg_off):
            if polypower == 0:
                where_chunk = np.where((x_knots[polypower]<=x)*(x<=x_knots[polypower+1]))
            else:
                where_chunk = np.where((x_knots[polypower]<x)*(x<=x_knots[polypower+1]))
            Moff[where_chunk[0],polypower] = 1-(x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
            Moff[where_chunk[0],1+polypower] = (x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
        # print(M.shape,Moff.shape)
        M = np.concatenate([M,Moff], axis = 1)

    where_data_finite = np.where(np.isfinite(spectrum))
    d = spectrum[where_data_finite]
    d_err = spec_err[where_data_finite]* sig
    M = M[where_data_finite[0],:]
    try:
        p,chi2,rank,s = np.linalg.lstsq(M/d_err[:,None],d/d_err,rcond=None)
        m=np.zeros(spectrum.shape)+np.nan
        m[where_data_finite] = np.dot(M,p)
        return m
    except:
        return None
    # res = d-m
    # print(p)
    #
    # plt.plot(x,m,label="model")
    # plt.plot(x,spectrum,label="data")
    # plt.fill_between(x,spectrum-spec_err,spectrum+spec_err,label="data err",alpha=0.5)
    # plt.legend()
    # plt.show()



def wavcal_nloglike_poly(paras, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitsrv,rv):
    m = wavcal_model_poly(paras, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitsrv,rv)
    if m is not None:
        nloglike = np.nansum((spectrum - m) ** 2 / (spec_err) ** 2)
        return 1 / 2. * nloglike
    else:
        return np.inf

def _fit_wavcal_poly(paras):
    x,wvs0,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitsrv,rv = paras
    dwv = 2*(wvs0[np.size(wvs0)//2]-wvs0[np.size(wvs0)//2-1])
    # print(dwv,0.3/2048)
    # exit()

    ## pre optimization with grid search and smaller dimensional space
    # dwv = 0.3 / 2048
    tmp_deg_wvs = 2
    N_dwv = 10
    # plt.plot(x,spectrum,label="data")
    # plt.fill_between(x,spectrum-spec_err,spectrum+spec_err,label="data err",alpha=0.5)
    # plt.plot(x,wavcal_model_poly([wvs0[0]-138*dwv/2,wvs0[np.size(wvs0)//2]-130*dwv/2,wvs0[-1]-125*dwv/2,5000], x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,tmp_deg_wvs,1,fitsrv,rv),label="model")
    # plt.legend()
    # plt.show()
    wvs_min = np.arange(wvs0[0]-N_dwv*dwv,wvs0[0]+N_dwv*dwv,dwv)#-130*dwv/2
    wvs_mid = np.arange(wvs0[2048//2]-N_dwv*dwv,wvs0[2048//2]+N_dwv*dwv,dwv)#-130*dwv/2
    wvs_max = np.arange(wvs0[-1]-N_dwv*dwv,wvs0[-1]+N_dwv*dwv,dwv)#-125*dwv/2
    nloglike_arr = np.zeros((np.size(wvs_min),np.size(wvs_mid),np.size(wvs_max)))
    for k,wv_min in enumerate(wvs_min):
        # print(k)
        for l, wv_mid in enumerate(wvs_mid):
            for m, wv_max in enumerate(wvs_max):
                nloglike_arr[k,l,m] = wavcal_nloglike_poly([wv_min,wv_mid,wv_max,5000], x, spectrum, spec_err,instr_trans, phoenix_HR8799_func, atran_2d_func, 2, deg_cont,False,rv)
    argmin2d = np.unravel_index(np.argmin(nloglike_arr),nloglike_arr.shape)
    M2 = np.zeros((np.size(x),(tmp_deg_wvs+1)))
    x_knots = x[np.linspace(0,len(x)-1,tmp_deg_wvs+1,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
    for polypower in range(tmp_deg_wvs):
        if polypower == 0:
            where_chunk = np.where((x_knots[polypower]<=x)*(x<=x_knots[polypower+1]))
        else:
            where_chunk = np.where((x_knots[polypower]<x)*(x<=x_knots[polypower+1]))
        M2[where_chunk[0],polypower] = 1-(x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
        M2[where_chunk[0],1+polypower] = (x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
    wvs2 = np.dot(M2,np.array([wvs_min[argmin2d[0]],wvs_mid[argmin2d[1]],wvs_max[argmin2d[2]]]))


    ## Real initialization
    x_knots = x[np.linspace(0, len(x) - 1, deg_wvs + 1, endpoint=True).astype(np.int)]
    if fitsrv:
        paras0 = np.array(wvs2[x_knots].tolist() + [rv,8000])
    else:
        paras0 = np.array(wvs2[x_knots].tolist() + [8000])
    simplex_init_steps =  np.ones(np.size(paras0))
    simplex_init_steps[0:deg_wvs+1] = dwv/4
    if fitsrv:
        simplex_init_steps[deg_wvs+1] = 1
        simplex_init_steps[deg_wvs+2] = 200
    else:
        simplex_init_steps[deg_wvs+1] = 200
    # print(paras0)
    # exit()
    initial_simplex = np.concatenate([paras0[None,:],paras0[None,:] + np.diag(simplex_init_steps)],axis=0)
    res = minimize(lambda paras: wavcal_nloglike_poly(paras, x,spectrum,spec_err, instr_trans,phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitsrv,rv), paras0, method="nelder-mead",
                           options={"xatol": 1e-10, "maxiter": 1e5,"initial_simplex":initial_simplex,"disp":False})
    out = res.x
    print(res.x)
    print(paras0)
    # print(out)
    # plt.plot(x,wavcal_model_poly(paras0, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont),label="model0")
    plt.plot(x,spectrum,label="data")
    plt.fill_between(x,spectrum-spec_err,spectrum+spec_err,label="data err",alpha=0.5)
    plt.plot(x,wavcal_model_poly(out, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitsrv,rv),label="model")
    plt.legend()
    # plt.figure(2)
    # out[-1] = 3000
    # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    # out[-1] = 4000
    # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    # out[-1] = 5000
    # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    # out[-1] = 8000
    # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    # plt.legend()
    plt.show()

    spl = InterpolatedUnivariateSpline(x_knots, out[0:deg_wvs+1], k=3, ext=0)
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
    phoenix_folder = "/scr3/jruffio/data/kpic/models/phoenix/"
    # mydir = os.path.join(mykpicdir, "20191107_kap_And")
    # target_rv = -12.70  # kap And
    # phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "kap_And" + "*.fits"))[0]
    mydir = os.path.join(mykpicdir, "20191108_bet_Peg")
    target_rv = 7.99  # bet Peg
    # mydir = os.path.join(mykpicdir,"20191215_HD_295747")
    # target_rv = 49.91 # bet Peg
    phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "bet_Peg" + "*.fits"))[0]
    # mydir = os.path.join(mykpicdir, "20191012_HD_1160_A")
    # target_rv = 12.60
    # phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "kap_And" + "*.fits"))[0]
    fitsrv = True


    mydate = os.path.basename(mydir).split("_")[0]

    if 0:
        hdulist = pyfits.open(os.path.join(mykpicdir,"special_calib", "wvs_f1.fits"))
        wvs_f0 = hdulist[0].data
        hdulist = pyfits.open(os.path.join(mykpicdir, "special_calib", "wvs_f2.fits"))
        wvs_f1 = hdulist[0].data
        hdulist = pyfits.open(os.path.join(mykpicdir, "special_calib", "wvs_f3.fits"))
        wvs_f2 = hdulist[0].data
        wvs = np.array([wvs_f0, wvs_f1, wvs_f2])
        print(wvs.shape)
    else:
        # hdulist = pyfits.open(os.path.join(mykpicdir, "special_calib", "20191215_kap_And_shifted_from_20191108_bet_Peg_wvs.fits"))
        hdulist = pyfits.open(os.path.join(mykpicdir, "special_calib", "20191107_kap_And"+"_wvs.fits"))
        wvs = hdulist[0].data

    # if 0: # test
    #     out = os.path.join(mydir, "calib", os.path.basename(mydir)+"_wvs.fits")
    #     hdulist = pyfits.open(out)
    #     wvs_new = hdulist[0].data
    #
    #     tmp = np.zeros((3,2048))
    #     for fib in range(3):
    #         for x in np.arange(0,2048,100):
    #             # plt.plot(wvs_new[fib,:,x],"x")
    #             p = np.polyfit([0,1,2,3,4,5,6,7,8],wvs_new[fib,[0,1,2,3,4,5,6,7,8],x],deg=1)
    #             # plt.plot(np.polyval(p,np.arange(9)))
    #
    #             plt.figure(2)
    #             plt.plot((wvs_new[fib,:,x]-np.polyval(p,np.arange(9)))/(wvs_new[fib,:,x+1]-wvs_new[fib,:,x]))
    #             plt.show()
    #             # f = interp1d([0,1,2,4,5,6,7,8],wvs_new[fib,[0,1,2,4,5,6,7,8],x])
    #             # tmp[fib,x] = f(3)
    #             # wvs_new[fib,3,x] = f(3)
    #     exit()
    #     plt.subplot(9,1,9-3)
    #     plt.plot(tmp[0,:]-wvs_new[0,3,:],color="orange",linestyle="--")
    #     plt.plot(tmp[1,:]-wvs_new[0,3,:],color="blue",linestyle="--")
    #     plt.plot(tmp[2,:]-wvs_new[0,3,:],color="purple",linestyle="--")
    #     for l in range(9):
    #         plt.subplot(9,1,9-l)
    #         plt.plot(wvs_new[0,l,:]-wvs_new[0,l,:],color="orange")
    #         plt.plot(wvs_new[1,l,:]-wvs_new[0,l,:],color="blue")
    #         plt.plot(wvs_new[2,l,:]-wvs_new[0,l,:],color="purple")
    #
    #         # plt.plot(wvs[0,l,:],color="orange",linestyle="--")
    #         # plt.plot(wvs[1,l,:],color="blue",linestyle="--")
    #         # plt.plot(wvs[2,l,:],color="purple",linestyle="--")
    #     plt.show()
    #
    #     exit()

    filelist = glob(os.path.join(mydir, "*fluxes.fits"))

    fluxes_list = []
    errors_list = []
    fiber_list = []
    header_list = []
    baryrv_list = []
    for filename in filelist:  #
        print(filename)
        hdulist = pyfits.open(filename)
        fluxes = hdulist[0].data
        header = hdulist[0].header
        errors = hdulist[1].data

        baryrv_list.append(float(header["BARYRV"]))
        fiber_list.append(np.argmax(np.nansum(fluxes, axis=(1, 2))))
        fluxes_list.append(fluxes)
        errors_list.append(errors)
        header_list.append(header)
    fiber_list = np.array(fiber_list)
    fluxes_list = np.array(fluxes_list)
    errors_list = np.array(errors_list)
    baryrv_list = np.array(baryrv_list)
    print(fluxes_list.shape)
    print(fiber_list)
    print(baryrv_list)
    # exit()

    combined_spec = np.zeros((3, 9, 2048))
    combined_spec_sig = np.zeros((3, 9, 2048))
    baryrv = np.zeros(3)
    for fib in np.unique(fiber_list):
        spec_list = fluxes_list[np.where(fib == fiber_list)[0], fib, :, :]
        spec_sig_list = errors_list[np.where(fib == fiber_list)[0], fib, :, :]
        baryrv[fib] = np.mean(baryrv_list[np.where(fib == fiber_list)[0]])
        # plt.figure(1)
        # print(fib,np.where(fib == fiber_list)[0])
        # for file_id in range(spec_list.shape[0]):
        #     for order_id in range(9):
        #         plt.subplot(9, 1, 9-order_id)
        #         plt.plot(spec_list[file_id,order_id,:],linestyle="-",linewidth=2,label="data {0}".format(file_id))
        #         plt.plot(spec_sig_list[file_id,order_id,:],linestyle="-",linewidth=2,label="error {0}".format(file_id))
        #         plt.legend()
        # plt.show()
        combined_spec[fib, :, :], combined_spec_sig[fib, :, :] = combine_stellar_spectra(spec_list, spec_sig_list)
    # exit()
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

    instr_trans_filename = glob(os.path.join(mydir, "calib", "*_instr_trans.fits"))[0]
    hdulist = pyfits.open(instr_trans_filename)
    instr_trans = hdulist[0].data

    line_width_func_list = []
    pixel_width_func_list = []
    line_width_filename = glob(os.path.join(mydir, "calib", "*_line_width_smooth.fits"))[0]
    hdulist = pyfits.open(line_width_filename)
    line_width = hdulist[0].data
    for fib in range(3):
        dwvs = wvs[fib][:, 1:2048] - wvs[fib][:, 0:2047]
        dwvs = np.concatenate([dwvs, dwvs[:, -1][:, None]], axis=1)
        line_width_wvunit = line_width[fib, :, :] * dwvs
        line_width_func = interp1d(np.ravel(wvs[fib]), np.ravel(line_width_wvunit), bounds_error=False,
                                   fill_value=np.nan)
        pixel_width_func = interp1d(np.ravel(wvs[fib]), np.ravel(dwvs), bounds_error=False, fill_value=np.nan)
        line_width_func_list.append(line_width_func)
        pixel_width_func_list.append(pixel_width_func)

    phoenix_folder = "/scr3/jruffio/data/kpic/models/phoenix/"
    phoenix_wv_filename = os.path.join(phoenix_folder, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
    with pyfits.open(phoenix_wv_filename) as hdulist:
        phoenix_wvs = hdulist[0].data / 1.e4
    crop_phoenix = np.where((phoenix_wvs > 1.8 - (2.6 - 1.8) / 2) * (phoenix_wvs < 2.6 + (2.6 - 1.8) / 2))
    phoenix_wvs = phoenix_wvs[crop_phoenix]
    with pyfits.open(phoenix_model_host_filename) as hdulist:
        phoenix_A0 = hdulist[0].data[crop_phoenix]

    new_wvs_arr = np.zeros((3,9,2048))

    for fib in np.unique(fiber_list):

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
            #l=2 end of the band is probably off
            #l=3, not sure
            l = 6
            x = np.arange(0,2048)
            spectrum = combined_spec[fib,l,:]
            spec_err = combined_spec_sig[fib,l,:]
            # spec_err = np.clip(spec_err,0.5*np.nanmedian(spec_err),np.inf)
            wvs0 = wvs[fib,l,:]
            deg_wvs = 5
            deg_cont= 5
            rv =  target_rv-baryrv[fib]
            print(rv)
            # exit()

            # hdulist = pyfits.open(glob(os.path.join(mydir, "calib", "*_wvs.fits"))[0])
            # wvs = hdulist[0].data
            # for k in range(9):
            #     plt.subplot(9,1,9-k)
            #     plt.plot(wvs[fib,k,:],combined_spec[fib,k,:],label="data")
            #     plt.fill_between(wvs[fib,k,:],combined_spec[fib,k,:]-combined_spec_sig[fib,k,:],combined_spec[fib,k,:]+combined_spec_sig[fib,k,:],label="data err",alpha=0.5)
            #     # plt.plot(host_spec[fib,k,:])
            # plt.show()

            # plt.fill_between(np.ravel(wvs0),np.ravel(spectrum-spec_err),np.ravel(spectrum+spec_err))
            # plt.show()

            new_wvs, out, x_knots = _fit_wavcal_poly((x, wvs0, spectrum, spec_err, instr_trans[fib,l, :],
                                                      phoenix_A0_func, atran_2d_func, deg_wvs, deg_cont, fitsrv,rv))
            print(out)
            exit()
        else:
            x = np.arange(0,2048)
            deg_wvs = 5
            deg_cont= 5
            rv =  target_rv-baryrv[fib]

            wvs0_chunks = []
            spectrum_chunks = []
            spec_err_chunks = []
            instr_trans_chunks = []
            kl_list = []
            for l in range(9):
                kl_list.append(l)
                wvs0_chunks.append(wvs[fib,l,:])
                spectrum_chunks.append(combined_spec[fib,l,:])
                instr_trans_chunks.append(instr_trans[fib,l, :])
                spec_err_chunks.append(combined_spec_sig[fib,l,:])
            outputs_list = mypool.map(_fit_wavcal_poly, zip(itertools.repeat(x),
                                                            wvs0_chunks, spectrum_chunks, spec_err_chunks,
                                                            instr_trans_chunks,
                                                            itertools.repeat(phoenix_A0_func),
                                                            itertools.repeat(atran_2d_func),
                                                            itertools.repeat(deg_wvs),
                                                            itertools.repeat(deg_cont),
                                                            itertools.repeat(fitsrv),
                                                            itertools.repeat(rv)))

            for kl, out in zip(kl_list, outputs_list):
                new_wvs, out, x_knots = out
                new_wvs_arr[fib,kl, :] = new_wvs

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=new_wvs_arr,header=header_list[0]))
    out = os.path.join(mydir, "calib", os.path.basename(mydir)+"_wvs.fits")
    print(out)
    try:
        hdulist.writeto(out, overwrite=True)
    except TypeError:
        hdulist.writeto(out, clobber=True)
    hdulist.close()
    exit()














    exit()
