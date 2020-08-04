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
    # print(star_rv)
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


def wavcal_jointorder_nloglike_poly(paras, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont):
    nloglike = 0
    # print(paras[-2])
    for orderid in range(spectrum.shape[0]):
        subparas = copy(paras[(deg_wvs+1)*orderid:(deg_wvs+1)*(orderid+1)])
        subparas = np.append(subparas,paras[-1])
        m = wavcal_model_poly(subparas, x,spectrum[orderid,:],spec_err[orderid,:],instr_trans[orderid,:], phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,False,paras[-2])
        if m is not None:
            nloglike += 1 / 2. * np.nansum((spectrum[orderid,:] - m) ** 2 / (spec_err[orderid,:]) ** 2)
        else:
            nloglike += np.inf
    return nloglike

def wavcal_nloglike_poly(paras, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitsrv,rv):
    m = wavcal_model_poly(paras, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitsrv,rv)
    if m is not None:
        nloglike = np.nansum((spectrum - m) ** 2 / (spec_err) ** 2)
        return 1 / 2. * nloglike
    else:
        return np.inf

def _fit_wavcal_grid(paras):
    x,wvs0,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,rv = paras
    tmp_deg_wvs = 3
    N_dwv = 10
    x_knots = x[np.linspace(0,len(x)-1,tmp_deg_wvs+1,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
    dwv = 2*(wvs0[np.size(x)//2]-wvs0[np.size(x)//2-1])

    wvs_min = np.arange(wvs0[0]-N_dwv*dwv,wvs0[0]+N_dwv*dwv,dwv)#-130*dwv/2
    wvs_mid = np.arange(wvs0[2048//3]-N_dwv*dwv,wvs0[2048//3]+N_dwv*dwv,dwv)#-130*dwv/2
    wvs_mid2 = np.arange(wvs0[2*2048//3]-N_dwv*dwv,wvs0[2*2048//3]+N_dwv*dwv,dwv)#-130*dwv/2
    wvs_max = np.arange(wvs0[-1]-N_dwv*dwv,wvs0[-1]+N_dwv*dwv,dwv)#-125*dwv/2
    nloglike_arr = np.zeros((np.size(wvs_min),np.size(wvs_mid),np.size(wvs_mid2),np.size(wvs_max)))
    for k,wv_min in enumerate(wvs_min):
        # print(k)
        for l, wv_mid in enumerate(wvs_mid):
            for l2, wv_mid2 in enumerate(wvs_mid2):
                for m, wv_max in enumerate(wvs_max):
                    nloglike_arr[k,l,l2,m] = wavcal_nloglike_poly([wv_min,wv_mid,wv_mid2,wv_max,5000], x, spectrum, spec_err,instr_trans, phoenix_HR8799_func, atran_2d_func, tmp_deg_wvs, deg_cont,False,rv)
    argmin2d = np.unravel_index(np.argmin(nloglike_arr),nloglike_arr.shape)

    spl = InterpolatedUnivariateSpline(x_knots, [wvs_min[argmin2d[0]], wvs_mid[argmin2d[1]], wvs_mid2[argmin2d[2]],
                                           wvs_max[argmin2d[3]]], k=3, ext=0)
    wvs2 = spl(x)


    ## Real initialization
    x_knots = x[np.linspace(0, len(x) - 1, deg_wvs + 1, endpoint=True).astype(np.int)]
    paras0 = np.array(wvs2[x_knots].tolist() + [8000])
    simplex_init_steps =  np.ones(np.size(paras0))
    simplex_init_steps[0:deg_wvs+1] = dwv/4
    simplex_init_steps[deg_wvs+1] = 200
    initial_simplex = np.concatenate([paras0[None,:],paras0[None,:] + np.diag(simplex_init_steps)],axis=0)
    res = minimize(lambda paras: wavcal_nloglike_poly(paras, x,spectrum,spec_err, instr_trans,phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,False,rv), paras0, method="nelder-mead",
                           options={"xatol": 1e-10, "maxiter": 1e5,"initial_simplex":initial_simplex,"disp":False})
    out = res.x
    spl = InterpolatedUnivariateSpline(x_knots, out[0:deg_wvs+1], k=3, ext=0)
    return spl(x)

def _fit_wavcal(paras):
    x,wvs0_arr,spectrum_arr,spec_err_arr,instr_trans_arr, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitsrv,rv,tpool = paras
    # print(dwv,0.3/2048)
    # exit()

    ## pre optimization with grid search and smaller dimensional space
    # dwv = 0.3 / 2048
    # plt.plot(x,spectrum,label="data")
    # plt.fill_between(x,spectrum-spec_err,spectrum+spec_err,label="data err",alpha=0.5)
    # plt.plot(x,wavcal_model_poly([wvs0[0]-138*dwv/2,wvs0[np.size(wvs0)//2]-130*dwv/2,wvs0[-1]-125*dwv/2,5000], x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,tmp_deg_wvs,1,fitsrv,rv),label="model")
    # plt.legend()
    # plt.show()
    wvs2_arr = np.zeros(wvs0_arr.shape)
    if 1:
        outputs_list = mypool.map(_fit_wavcal_grid, zip(itertools.repeat(x),
                                                        wvs0_arr,spectrum_arr,spec_err_arr,instr_trans_arr,
                                                        itertools.repeat(phoenix_A0_func),
                                                        itertools.repeat(atran_2d_func),
                                                        itertools.repeat(deg_cont),
                                                        itertools.repeat(deg_wvs),
                                                        itertools.repeat(rv)))

        for orderid, out in enumerate(outputs_list):
            wvs2_arr[orderid, :] = out
    else:
        for orderid,(wvs0, spectrum, spec_err, instr_trans) in enumerate(zip(wvs0_arr, spectrum_arr, spec_err_arr, instr_trans_arr)):
            print(x,wvs0,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_cont,rv)
            wvs2_arr[orderid,:] = _fit_wavcal_grid((x,wvs0,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,rv))
            # wvs_min = np.arange(wvs0[0]-N_dwv*dwv,wvs0[0]+N_dwv*dwv,dwv)#-130*dwv/2
            # wvs_mid = np.arange(wvs0[2048//3]-N_dwv*dwv,wvs0[2048//3]+N_dwv*dwv,dwv)#-130*dwv/2
            # wvs_mid2 = np.arange(wvs0[2*2048//3]-N_dwv*dwv,wvs0[2*2048//3]+N_dwv*dwv,dwv)#-130*dwv/2
            # wvs_max = np.arange(wvs0[-1]-N_dwv*dwv,wvs0[-1]+N_dwv*dwv,dwv)#-125*dwv/2
            # nloglike_arr = np.zeros((np.size(wvs_min),np.size(wvs_mid),np.size(wvs_mid2),np.size(wvs_max)))
            # for k,wv_min in enumerate(wvs_min):
            #     # print(k)
            #     for l, wv_mid in enumerate(wvs_mid):
            #         for l2, wv_mid2 in enumerate(wvs_mid2):
            #             for m, wv_max in enumerate(wvs_max):
            #                 nloglike_arr[k,l,l2,m] = wavcal_nloglike_poly([wv_min,wv_mid,wv_mid2,wv_max,5000], x, spectrum, spec_err,instr_trans, phoenix_HR8799_func, atran_2d_func, tmp_deg_wvs, deg_cont,False,rv)
            # argmin2d = np.unravel_index(np.argmin(nloglike_arr),nloglike_arr.shape)
            #
            # spl = InterpolatedUnivariateSpline(x_knots, [wvs_min[argmin2d[0]],wvs_mid[argmin2d[1]],wvs_mid2[argmin2d[2]],wvs_max[argmin2d[3]]], k=3, ext=0)
            # wvs2_arr[orderid,:] = spl(x)

    # for orderid in range(wvs0_arr.shape[0]):
    #     plt.subplot(3,1,orderid+1)
    #     plt.plot(wvs0_arr[orderid,:],wvs2_arr[orderid,:]-wvs0_arr[orderid,:])
    # plt.show()

    ## Real initialization
    x_knots = x[np.linspace(0, len(x) - 1, deg_wvs + 1, endpoint=True).astype(np.int)]
    paras0 = []
    for orderid in range(wvs2_arr.shape[0]):
        paras0 = paras0+wvs2_arr[orderid,:][x_knots].tolist()
    if fitsrv:
        paras0 = np.array(paras0 + [rv,8000])
    else:
        paras0 = np.array(paras0 + [8000])
    simplex_init_steps =  np.ones(np.size(paras0))
    for orderid in range(wvs2_arr.shape[0]):
        simplex_init_steps[orderid*(deg_wvs+1):(1+orderid)*(deg_wvs+1)] = (wvs2_arr[orderid,np.size(x)//2]-wvs2_arr[orderid,np.size(x)//2-1])
    if fitsrv:
        simplex_init_steps[wvs2_arr.shape[0]*(deg_wvs+1)] = 1
        simplex_init_steps[wvs2_arr.shape[0]*(deg_wvs+1)+1] = 200
    else:
        simplex_init_steps[wvs2_arr.shape[0]*(deg_wvs+1)] = 200
    print(paras0)
    # subparas = paras0[(deg_wvs+1)*orderid:(deg_wvs+1)*(orderid+1)]
    # subparas = np.append(subparas,paras0[-1])
    # print(subparas)
    # exit()
    initial_simplex = np.concatenate([paras0[None,:],paras0[None,:] + np.diag(simplex_init_steps)],axis=0)
    res = minimize(lambda paras: wavcal_jointorder_nloglike_poly(paras, x,spectrum_arr,spec_err_arr, instr_trans_arr,phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont), paras0, method="nelder-mead",
                           options={"xatol": 1e-10, "maxiter": 1e5,"initial_simplex":initial_simplex,"disp":False})
    out = res.x
    print(res)
    print(res.x)
    print(paras0)
    # print(out)
    wvs_out = np.zeros(wvs0_arr.shape)
    for orderid in range(wvs0_arr.shape[0]):
        spl = InterpolatedUnivariateSpline(x_knots, out[orderid*(deg_wvs+1):(orderid+1)*(deg_wvs+1)], k=3, ext=0)
        wvs_out[orderid,:] = spl(x)

    # tilderv = out[-2]
    # for orderid,(spectrum, spec_err, instr_trans) in enumerate(zip( spectrum_arr, spec_err_arr, instr_trans_arr)):
    #     print(orderid)
    #     plt.figure(1)
    #     subparas = out[(deg_wvs+1)*orderid:(deg_wvs+1)*(orderid+1)]
    #     subparas = np.append(subparas,out[-1])
    #     plt.subplot(spectrum_arr.shape[0],1,orderid+1)
    #     # plt.plot(x,wavcal_model_poly(paras0, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont),label="model0")
    #     plt.plot(x,spectrum,label="data")
    #     plt.fill_between(x,spectrum-spec_err,spectrum+spec_err,label="data err",alpha=0.5)
    #     plt.plot(x,wavcal_model_poly(subparas, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,False,tilderv),label="model")
    #     plt.scatter(x_knots,np.ones(x_knots.shape)*np.nanmedian(spectrum),c="red")
    #     plt.legend()
    #     plt.figure(2)
    #     plt.subplot(spectrum_arr.shape[0],1,orderid+1)
    #     plt.scatter(x_knots,np.ones(x_knots.shape)*np.nanmedian(wvs_out[orderid,:]),c="red")
    #     plt.plot(x,wvs_out[orderid,:],c="red")
    #     # out[-1] = 3000
    #     # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    #     # out[-1] = 4000
    #     # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    #     # out[-1] = 5000
    #     # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    #     # out[-1] = 8000
    #     # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    #     # plt.legend()
    # plt.show()

    return wvs_out,out,x_knots

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
    # target_rv = 49.91 # HD_295747
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

    filelist = glob(os.path.join(mydir, "*fluxes.fits"))

    if 1:
        wvs_list =[]
        rv_list = []
        fib_list = []
        for filename in filelist:
            print(filename)
            hdulist = pyfits.open(filename)
            fluxes = hdulist[0].data
            header = hdulist[0].header
            errors = hdulist[1].data

            baryrv = float(header["BARYRV"])
            fib = np.argmax(np.nansum(fluxes, axis=(1, 2)))

            out = os.path.join(mydir, "calib", os.path.basename(filename).replace(".fits", "") + "_wvs.fits")
            hdulist = pyfits.open(out)
            new_wvs = hdulist[0].data[fib,6,:]
            header = hdulist[0].header
            wvs_list.append(new_wvs)
            rv_list.append(float(header["WAVCALRV"]))
            fib_list.append(fib)
        wvs_list = np.array(wvs_list)
        rv_list = np.array(rv_list)
        med_wvs = np.nanmedian(wvs_list,axis=0)
        dwvs = wvs[fib,6,1::]-wvs[fib,6,0:np.size(med_wvs)-1]
        dwvs = np.append(dwvs,dwvs[-1])
        dev_list = np.nanmedian((wvs_list-wvs[fib,6,:][None,:])/dwvs[None,:],axis=1)
        print(dev_list)
        print(rv_list)
        linestyle_list = ["-","--",":"]
        color_list = ["#ff9900", "#0099cc", "#6600ff"]
        for linestyle,color,fib in zip(linestyle_list,color_list,np.arange(3)):
            where_fib = np.where(fib==fib_list)
            plt.subplot(2,1,1)
            plt.scatter(np.arange(np.size(rv_list))[where_fib],dev_list[where_fib],c=color)
            plt.subplot(2,1,2)
            plt.scatter(np.arange(np.size(rv_list))[where_fib],rv_list[where_fib],c=color,label="Fiber {0}".format(fib))
        plt.subplot(2,1,1)
        plt.ylabel("Order 6 - med. deviation (pix)")
        plt.subplot(2,1,2)
        plt.ylabel("RV bet Peg (km/s)")
        plt.fill_between(np.arange(np.size(rv_list)),(7.99-0.23)*np.ones(np.size(rv_list)),(7.99+0.23)*np.ones(np.size(rv_list)),
                         color="gray",alpha = 0.5)
        plt.legend()
        xticks_list = [os.path.basename(filename).replace("nspec","").replace("_fluxes.fits","") for filename in filelist]
        plt.xticks(np.arange(len(filelist)),xticks_list, rotation=90)
        plt.tight_layout()
        plt.show()

    for filename in filelist:  #
        print(filename)
        hdulist = pyfits.open(filename)
        fluxes = hdulist[0].data
        header = hdulist[0].header
        errors = hdulist[1].data

        baryrv = float(header["BARYRV"])
        fib = np.argmax(np.nansum(fluxes, axis=(1, 2)))

        combined_spec = edges2nans(fluxes)
        combined_spec_sig = edges2nans(errors)


        instr_trans_filename = glob(os.path.join(mydir, "calib", "*_instr_trans.fits"))[0]
        hdulist = pyfits.open(instr_trans_filename)
        instr_trans = hdulist[0].data

        line_width_filename = glob(os.path.join(mydir, "calib", "*_line_width_smooth.fits"))[0]
        hdulist = pyfits.open(line_width_filename)
        line_width = hdulist[0].data
        dwvs = wvs[fib][:, 1:2048] - wvs[fib][:, 0:2047]
        dwvs = np.concatenate([dwvs, dwvs[:, -1][:, None]], axis=1)
        line_width_wvunit = line_width[fib, :, :] * dwvs
        line_width_func = interp1d(np.ravel(wvs[fib]), np.ravel(line_width_wvunit), bounds_error=False,
                                   fill_value=np.nan)
        pixel_width_func = interp1d(np.ravel(wvs[fib]), np.ravel(dwvs), bounds_error=False, fill_value=np.nan)

        phoenix_folder = "/scr3/jruffio/data/kpic/models/phoenix/"
        phoenix_wv_filename = os.path.join(phoenix_folder, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
        with pyfits.open(phoenix_wv_filename) as hdulist:
            phoenix_wvs = hdulist[0].data / 1.e4
        crop_phoenix = np.where((phoenix_wvs > 1.8 - (2.6 - 1.8) / 2) * (phoenix_wvs < 2.6 + (2.6 - 1.8) / 2))
        phoenix_wvs = phoenix_wvs[crop_phoenix]
        with pyfits.open(phoenix_model_host_filename) as hdulist:
            phoenix_A0 = hdulist[0].data[crop_phoenix]

        new_wvs_arr = np.zeros((3,9,2048))

        phoenix_line_widths = np.array(pd.DataFrame(line_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
        phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
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
            x = np.arange(0,2048)
            selec_orders = [0,1,6,7]#3,4,5
            spectrum = combined_spec[fib,selec_orders,:]
            spec_err = combined_spec_sig[fib,selec_orders,:]
            # spec_err = np.clip(spec_err,0.5*np.nanmedian(spec_err),np.inf)
            wvs0 = wvs[fib,selec_orders,:]
            deg_wvs = 5
            deg_cont= 5
            rv =  target_rv-baryrv
            print(rv)

            new_wvs, out, x_knots = _fit_wavcal((x, wvs0, spectrum, spec_err, instr_trans[fib,selec_orders, :],
                                                      phoenix_A0_func, atran_2d_func, deg_wvs, deg_cont, fitsrv,rv,mypool))
            new_wvs_arr[fib, selec_orders,:] = new_wvs

        hdulist = pyfits.HDUList()
        header2save = copy(header)
        header2save["WAVCALWA"] = out[-1]
        header2save["WAVCALRV"] = out[-2]+baryrv
        print(header2save["WAVCALRV"])
        hdulist.append(pyfits.PrimaryHDU(data=new_wvs_arr,header=header2save))
        out = os.path.join(mydir, "calib", os.path.basename(filename).replace(".fits","")+"_wvs.fits")
        print(out)
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()
    exit()














    exit()
