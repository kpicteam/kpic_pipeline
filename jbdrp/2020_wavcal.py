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
from utils.badpix import *
from utils.misc import *
from utils.spectra import *

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
    x,wvs0,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitsrv,rv,debug = paras
    dwv = 3*(wvs0[np.size(wvs0)//2]-wvs0[np.size(wvs0)//2-1])
    spec_err = np.clip(spec_err,0.1*np.nanmedian(spec_err),np.inf)
    # print(dwv,0.3/2048)
    # exit()

    if 0:
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
        wvs_max = np.arange(wvs0[-1]-2*N_dwv*dwv,wvs0[-1]+2*N_dwv*dwv,dwv)#-125*dwv/2
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
    else:
        tmp_deg_wvs = 3
        N_dwv = 5
        x_knots = x[np.linspace(0,len(x)-1,tmp_deg_wvs+1,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )

        wvs_min = np.arange(wvs0[0]-N_dwv*dwv,wvs0[0]+N_dwv*dwv,dwv)#-130*dwv/2
        wvs_mid = np.arange(wvs0[2048//3]-N_dwv*dwv,wvs0[2048//3]+N_dwv*dwv,dwv)#-130*dwv/2
        wvs_mid2 = np.arange(wvs0[2*2048//3]-N_dwv*dwv,wvs0[2*2048//3]+N_dwv*dwv,dwv)#-130*dwv/2
        wvs_max = np.arange(wvs0[-1]-N_dwv*dwv,wvs0[-1]+N_dwv*dwv,dwv)#-125*dwv/2
        nloglike_arr = np.zeros((np.size(wvs_min),np.size(wvs_mid),np.size(wvs_mid2),np.size(wvs_max)))
        for k,wv_min in enumerate(wvs_min):
            print(k)
            for l, wv_mid in enumerate(wvs_mid):
                for l2, wv_mid2 in enumerate(wvs_mid2):
                    for m, wv_max in enumerate(wvs_max):
                        nloglike_arr[k,l,l2,m] = wavcal_nloglike_poly([wv_min,wv_mid,wv_mid2,wv_max,1000], x, spectrum, spec_err,instr_trans, phoenix_HR8799_func, atran_2d_func, tmp_deg_wvs, deg_cont,False,rv)
        argmin2d = np.unravel_index(np.argmin(nloglike_arr),nloglike_arr.shape)

        spl = InterpolatedUnivariateSpline(x_knots, [wvs_min[argmin2d[0]], wvs_mid[argmin2d[1]], wvs_mid2[argmin2d[2]],
                                               wvs_max[argmin2d[3]]], k=3, ext=0)
        wvs2 = spl(x)



    ## Real initialization
    x_knots = x[np.linspace(0, len(x) - 1, deg_wvs + 1, endpoint=True).astype(np.int)]
    if fitsrv:
        paras0 = np.array(wvs2[x_knots].tolist() + [rv,1000])
    else:
        paras0 = np.array(wvs2[x_knots].tolist() + [1000])
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

    if debug:
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

    numthreads = 10
    mypool = mp.Pool(processes=numthreads)

    mykpicdir = "/scr3/jruffio/data/kpic/"
    phoenix_folder = "/scr3/jruffio/data/kpic/models/phoenix/"
    # mydir = os.path.join(mykpicdir, "20191107_kap_And")
    # target_rv = -12.70  # kap And
    # phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "kap_And" + "*.fits"))[0]
    # mydir = os.path.join(mykpicdir, "20191108_bet_Peg")
    # target_rv = 7.99  # bet Peg
    # mydir = os.path.join(mykpicdir,"20191215_HD_295747")
    # target_rv = 49.91 # bet Peg
    # mydir = os.path.join(mykpicdir, "20200607_HIP_81497")
    # mydir = os.path.join(mykpicdir, "20200608_HIP_81497_30s")
    # mydir = os.path.join(mykpicdir, "20200608_HIP_81497_7.5s")
    # mydir = os.path.join(mykpicdir,"20200609_HIP_81497")
    # mydir = os.path.join(mykpicdir, "20200701_HIP_81497")
    # mydir = os.path.join(mykpicdir, "20200702_HIP_81497")
    # mydir = os.path.join(mykpicdir, "20200703_HIP_81497")
    # target_rv = -55.567  # bet Peg
    # phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "HIP_81497_lte03600-1.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"))[0]
    # mydir = os.path.join(mykpicdir,"20200928_HIP_95771")
    # mydir = os.path.join(mykpicdir,"20200929_HIP_95771")
    mydir = os.path.join(mykpicdir,"20201001_HIP_95771")
    target_rv = -85.391  # bet Peg
    phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "HIP_81497_lte03600-1.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"))[0]
    # mydir = os.path.join(mykpicdir, "20191012_HD_1160_A")
    # target_rv = 12.60
    # phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "kap_And" + "*.fits"))[0]
    fitsrv = False
    debug = True


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
        # _wvs = hdulist[0].data
        # wvs = np.zeros((_wvs.shape[0]+1,_wvs.shape[1],_wvs.shape[2]))
        # wvs[0:3,:,:] = _wvs
        # wvs[-1,:,:] = _wvs[-1,:,:]
        # hdulist = pyfits.open(os.path.join(mykpicdir, "special_calib", "20200607_HIP_81497_wvs.fits"))
        hdulist = pyfits.open(os.path.join(mykpicdir, "20200607_HIP_81497","calib", "20200607_HIP_81497_wvs.fits"))
        wvs = hdulist[0].data
        # wvs[2,4,:] = (wvs[1,4,:]+wvs[3,4,:])/2
        # wvs[3,2,:] = wvs[2,2,:]
        # wvs[:,3,:] = wvs[1,3,:][None,:]
        # wvs[:,3,:] = (2*wvs[:,2 ,:]+1*wvs[:,5,:])/3

    if 0: # test
        hdulist = pyfits.open(os.path.join(mykpicdir, "20200607_HIP_81497","calib", "20200607_HIP_81497_wvs.fits"))
        wvs = hdulist[0].data
        hdulist = pyfits.open(os.path.join(mykpicdir, "20200608_HIP_81497_30s","calib", "20200608_HIP_81497_30s_wvs.fits"))
        wvs2 = hdulist[0].data
        hdulist = pyfits.open(os.path.join(mykpicdir, "20200608_HIP_81497_7.5s","calib","archive", "20200608_HIP_81497_7.5s_wvs.fits"))
        wvs3 = hdulist[0].data
        hdulist = pyfits.open(os.path.join(mykpicdir, "20200609_HIP_81497","calib", "20200609_HIP_81497_wvs.fits"))
        wvs4 = hdulist[0].data
        line_width_filename = glob(os.path.join(mydir, "calib", "*_line_width_smooth.fits"))[0]
        hdulist = pyfits.open(line_width_filename)
        line_width = hdulist[0].data
        pixel_width = np.zeros(line_width.shape)
        line_width_wvunit = np.zeros(line_width.shape)
        for fib in range(wvs.shape[0]):
            dwvs = wvs[fib][:, 1:2048] - wvs[fib][:, 0:2047]
            dwvs = np.concatenate([dwvs, dwvs[:, -1][:, None]], axis=1)
            pixel_width[fib, :, :] = dwvs
            line_width_wvunit[fib, :, :]= line_width[fib, :, :] * dwvs

        # for fib in range(wvs.shape[0]):
        #     plt.subplot(4,1,fib+1)
        #     plt.title("fiber {0}".format(fib))
        #     for l in range(9):
        #         plt.plot(wvs2[fib,l,:]/line_width_wvunit[fib,l,:]/2.35,label="{0}".format(l))
        #         plt.plot(wvs3[fib,l,:]/line_width_wvunit[fib,l,:]/2.35,"--",label="{0}".format(l))
        # plt.show()

        plt.figure(2)
        for fib in [1]:#range(wvs.shape[0]):
            # plt.subplot(4,1,fib+1)
            for l in range(9):
                if l == 0:
                    plt.plot(np.ravel(wvs2[fib,l,:]),np.ravel((wvs2[fib,l,:]-wvs[fib,l,:])/pixel_width[fib,l,:]),label="June 7th - June 8th (early night)",color="orange")
                    plt.plot(np.ravel(wvs2[fib,l,:]),np.ravel((wvs3[fib,l,:]-wvs[fib,l,:])/pixel_width[fib,l,:]),label="June 7th - June 8th (later)",color="blue")
                    plt.plot(np.ravel(wvs2[fib,l,:]),np.ravel((wvs4[fib,l,:]-wvs[fib,l,:])/pixel_width[fib,l,:]),label="June 7th - June 9th",color="purple")
                else:
                    plt.plot(np.ravel(wvs2[fib,l,:]),np.ravel((wvs2[fib,l,:]-wvs[fib,l,:])/pixel_width[fib,l,:]),color="orange")
                    plt.plot(np.ravel(wvs2[fib,l,:]),np.ravel((wvs3[fib,l,:]-wvs[fib,l,:])/pixel_width[fib,l,:]),color="blue")
                    plt.plot(np.ravel(wvs2[fib,l,:]),np.ravel((wvs4[fib,l,:]-wvs[fib,l,:])/pixel_width[fib,l,:]),color="purple")
            plt.legend()
            plt.ylabel("pixels shift")
            plt.xlabel(r"$\lambda$ ($\mu$m)")
        plt.show()


        delta_wvs = wvs-wvs[:,0,:][:,None,:]
        delta_wvs = delta_wvs - delta_wvs[:,:,1000][:,:,None]
        plt.figure(1)
        for fib in range(wvs.shape[0]):
            plt.subplot(4,1,fib+1)
            for l in range(9):
                p = np.polyfit(np.arange(2048),delta_wvs[fib, l, :],deg=2)
                plt.plot(delta_wvs[fib,l,:]-np.polyval(p, np.arange(2048)),label="{0}".format(l))#
                plt.ylim([-0.0001,0.0001])
            plt.legend()
        # plt.figure(2)
        # for fib in range(wvs.shape[0]):
        #     for l in range(9):
        #         p = np.polyfit(np.arange(2048),delta_wvs[fib, l, :],deg=2)
        #         for ip,_p in enumerate(p):
        #             plt.subplot(4,3,(fib*3)+1+ip)
        #             plt.scatter(l,_p)
        plt.show()

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

    combined_spec = np.zeros(wvs.shape)
    combined_spec_sig = np.zeros(wvs.shape)
    baryrv = np.zeros(wvs.shape[0])
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
    for fib in range(wvs.shape[0]):
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


    if 0:
        for fib in range(4):
            atran_func_list = []
            # atran_filename = os.path.join("/scr3/jruffio/data/kpic/","models","atran","atran_13599_30_0_2_45_135_245_0.txt")
            atran_filelist = glob(os.path.join("/scr3/jruffio/data/kpic/","models","atran","atran_13599_30_*.dat"))
            atran_filelist.sort()
            print(atran_filelist)
            water_list = np.array([int(atran_filename.split("_")[-5]) for atran_filename in atran_filelist])
            waterargsort = np.argsort(water_list).astype(np.int)
            water_list = np.array([water_list[k] for k in waterargsort])
            atran_filelist = [atran_filelist[k] for k in waterargsort]
            atran_spec_list = []
            for k,atran_filename in enumerate(atran_filelist):
                print(atran_filename)
                atran_arr = np.loadtxt(atran_filename).T
                atran_wvs = atran_arr[1,:]
                atran_spec = atran_arr[2,:]
                atran_line_widths = np.array(pd.DataFrame(line_width_func_list[fib](atran_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                atran_pixel_widths = np.array(pd.DataFrame(pixel_width_func_list[fib](atran_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                atran_spec_conv = convolve_spectrum_line_width(atran_wvs,atran_spec,atran_line_widths,mypool=mypool)
                atran_spec_conv2 = convolve_spectrum_pixel_width(atran_wvs,atran_spec_conv,atran_pixel_widths,mypool=mypool)
                atran_spec_list.append(atran_spec_conv2)
                # plt.plot(atran_wvs,atran_spec,label=os.path.basename(atran_filename))
                # plt.plot(atran_wvs,atran_spec_conv,label=os.path.basename(atran_filename))
                plt.plot(atran_wvs,atran_spec_conv2,label=os.path.basename(atran_filename))
                # plt.show()

            # print(np.array(atran_spec_list).shape)
            # print(water_list.shape,atran_wvs.shape)
            # atran_2d_func = interp2d(atran_wvs,water_list, np.array(atran_spec_list), bounds_error=False, fill_value=np.nan)
            # # print(atran_2d_func(wvs_per_orders[3,:],1.5*np.ones(wvs_per_orders[3,:].shape)).shape)
            # plt.plot(wvs_per_orders[3,:],atran_2d_func(wvs_per_orders[3,:],2000),label="interp")
            plt.legend()

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=np.array(atran_spec_list)))
            out = os.path.join(mykpicdir, "models", "atran", "2020_atran_spec_list_f{0}.fits".format(fib))
            # out = os.path.join(kap_And_dir, "..","models","atran", "atran_spec_list.fits")
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=np.array(water_list)))
            out = os.path.join(mykpicdir,"models","atran", "2020_water_list.fits")
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=np.array(atran_wvs)))
            out = os.path.join(mykpicdir,"models","atran", "2020_atran_wvs.fits")
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()
        plt.show()
        exit()


    new_wvs_arr = np.zeros(wvs.shape)

    for fib in np.unique(fiber_list):

        phoenix_line_widths = np.array(pd.DataFrame(line_width_func_list[fib](phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
        # phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func_list[fib](phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
        phoenix_A0_conv = convolve_spectrum_line_width(phoenix_wvs, phoenix_A0, phoenix_line_widths, mypool=mypool)
        # phoenix_A0_conv = convolve_spectrum_pixel_width(phoenix_wvs, phoenix_A0_conv, phoenix_pixel_widths,mypool=mypool)
        phoenix_A0_func = interp1d(phoenix_wvs, phoenix_A0_conv / np.nanmax(phoenix_A0_conv), bounds_error=False,fill_value=np.nan)

        out = os.path.join(mydir, "..", "models", "atran", "2020_atran_spec_list_f{0}.fits".format(fib))
        hdulist = pyfits.open(out)
        atran_spec_list = hdulist[0].data
        out = os.path.join(mydir, "..", "models", "atran", "2020_water_list.fits")
        hdulist = pyfits.open(out)
        water_list = hdulist[0].data
        out = os.path.join(mydir, "..", "models", "atran", "2020_atran_wvs.fits")
        hdulist = pyfits.open(out)
        atran_wvs = hdulist[0].data
        atran_2d_func = interp2d(atran_wvs, water_list, np.array(atran_spec_list), bounds_error=False,fill_value=np.nan)

        if debug: # jump
            #l=2 end of the band is probably off
            #l=3, not sure
            l = 6

            x = np.arange(0,2048)
            spectrum = combined_spec[fib,l,:]
            spec_err = combined_spec_sig[fib,l,:]
            spec_err = np.clip(spec_err,0.1*np.nanmedian(spec_err),np.inf)
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
                                                      phoenix_A0_func, atran_2d_func, deg_wvs, deg_cont, fitsrv,rv,debug))
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
                                                            itertools.repeat(rv),
                                                            itertools.repeat(debug)))

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
