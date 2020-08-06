import numpy as np
from copy import copy
import astropy.io.fits as fits
from scipy.interpolate import interp1d
import itertools
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
from kpicdrp.utils import *

def edges2nans(spec):
    """
    Set edges to Nans. Hard coded.
    """
    cp_spec = copy(spec)
    if np.size(spec.shape) == 3:
        cp_spec[:,:,0:5] = np.nan
        cp_spec[:,:,2000::] = np.nan
    elif np.size(spec.shape) == 2:
        cp_spec[:,0:5] = np.nan
        cp_spec[:,2000::] = np.nan
    elif np.size(spec.shape) == 1:
        cp_spec[0:5] = np.nan
        cp_spec[2000::] = np.nan
    return cp_spec

def save_atrangrid(filelist_atran,line_width_func,atrangridname,mypool=None):
    """
    Broaden and save a grid of telluric models using atran.
    """
    filelist_atran.sort() #'/scr3/jruffio/data/kpic/models/atran/atran_13599_30_500_2_0_1.9_2.6.dat'
    water_list = np.array([int(atran_filename.split("_")[-5]) for atran_filename in filelist_atran])
    angle_list = np.array([float(atran_filename.split("_")[-3]) for atran_filename in filelist_atran])
    water_unique = np.unique(water_list)
    angle_unique = np.unique(angle_list)

    atran_spec_list = []
    for water in water_unique:
        for angle in angle_unique:
            print(water,angle)
            atran_filename = filelist_atran[np.where((water==water_list)*(angle==angle_list))[0][0]]
            atran_arr = np.loadtxt(atran_filename).T
            atran_wvs = atran_arr[1,:]
            atran_spec = atran_arr[2,:]
            atran_line_widths = np.array(pd.DataFrame(line_width_func(atran_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
            atran_spec_conv = convolve_spectrum_line_width(atran_wvs,atran_spec,atran_line_widths,mypool=mypool)
            atran_spec_list.append(atran_spec_conv)

    atran_grid = np.zeros((np.size(water_unique),np.size(angle_unique),np.size(atran_wvs)))
    for water_id,water in enumerate(water_unique):
        for angle_id,angle in enumerate(angle_unique):
            atran_grid[water_id,angle_id,:] = atran_spec_list[np.where((water_list==water)*(angle_list==angle))[0][0]]

    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU(data=atran_grid))
    hdulist.append(fits.ImageHDU(data=water_unique))
    hdulist.append(fits.ImageHDU(data=angle_unique))
    hdulist.append(fits.ImageHDU(data=atran_wvs))
    try:
        hdulist.writeto(atrangridname, overwrite=True)
    except TypeError:
        hdulist.writeto(atrangridname, clobber=True)
    hdulist.close()

def wavcal_model(paras, x,spectrum,spec_err, star_func,telluric_wvs,telluric_interpgrid,N_nodes_wvs,blaze_chunks,fitsrv,rv):
    c_kms = 299792.458
    wvs_coefs = paras[0:N_nodes_wvs]
    sig = 1
    if fitsrv:
        star_rv = paras[N_nodes_wvs]
        water = paras[N_nodes_wvs+1]
        angle = paras[N_nodes_wvs+2]
    else:
        star_rv = rv
        water = paras[N_nodes_wvs]
        angle = paras[N_nodes_wvs+1]

    tel_func = interp1d(telluric_wvs,telluric_interpgrid([water,angle])[0],bounds_error=False,fill_value=0)

    if N_nodes_wvs <=3:
        M2 = np.zeros((np.size(x),(N_nodes_wvs)))
        x_knots = x[np.linspace(0,len(x)-1,N_nodes_wvs,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
        for polypower in range(N_nodes_wvs-1):
            if polypower == 0:
                where_chunk = np.where((x_knots[polypower]<=x)*(x<=x_knots[polypower+1]))
            else:
                where_chunk = np.where((x_knots[polypower]<x)*(x<=x_knots[polypower+1]))
            M2[where_chunk[0],polypower] = 1-(x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
            M2[where_chunk[0],1+polypower] = (x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
        wvs = np.dot(M2,wvs_coefs)
    else:
        x_knots = x[np.linspace(0,len(x)-1,N_nodes_wvs,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
        spl = InterpolatedUnivariateSpline(x_knots,wvs_coefs,k=3,ext=0)
        wvs= spl(x)

    M = np.zeros((np.size(x),(blaze_chunks+1)))
    tmp = star_func(wvs*(1-star_rv/c_kms))*tel_func(wvs)
    M0_mn = np.nanmean(tmp)
    tmp /= M0_mn
    x_knots = x[np.linspace(0,len(x)-1,blaze_chunks+1,endpoint=True).astype(np.int)]
    for polypower in range(blaze_chunks):
        if polypower == 0:
            where_chunk = np.where((x_knots[polypower]<=x)*(x<=x_knots[polypower+1]))
        else:
            where_chunk = np.where((x_knots[polypower]<x)*(x<=x_knots[polypower+1]))
        M[where_chunk[0],polypower] = 1-(x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
        M[where_chunk[0],1+polypower] = (x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
    M = tmp[:,None]*M

    if 0:
        deg_off = 1
        Moff = np.zeros((np.size(x),(deg_off+1)))
        x_knots = x[np.linspace(0,len(x)-1,deg_off+1,endpoint=True).astype(np.int)]
        for polypower in range(deg_off):
            if polypower == 0:
                where_chunk = np.where((x_knots[polypower]<=x)*(x<=x_knots[polypower+1]))
            else:
                where_chunk = np.where((x_knots[polypower]<x)*(x<=x_knots[polypower+1]))
            Moff[where_chunk[0],polypower] = 1-(x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
            Moff[where_chunk[0],1+polypower] = (x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
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

def wavcal_nloglike(paras, x,spectrum,spec_err, star_func,telluric_wvs,telluric_interpgrid,N_nodes_wvs,blaze_chunks,fitsrv,rv):
    m = wavcal_model(paras, x,spectrum,spec_err, star_func,telluric_wvs,telluric_interpgrid,N_nodes_wvs,blaze_chunks,fitsrv,rv)
    if m is not None:
        nloglike = np.nansum((spectrum - m) ** 2 / (spec_err) ** 2)
        return 1 / 2. * nloglike
    else:
        return np.inf

def _fit_wavecal(paras):
    x,wvs0,spectrum,spec_err, star_func,telluric_wvs,telluric_interpgrid,N_nodes_wvs,blaze_chunks,fitsrv,rv,\
        init_grid_search,init_grid_dwv = paras
    dwv = 3*(wvs0[np.size(wvs0)//2]-wvs0[np.size(wvs0)//2-1])

    ## pre optimization with grid search and smaller dimensional space
    if init_grid_search:
        tmp_deg_wvs = 4
        N_dwv = init_grid_dwv//dwv
        x_knots = x[np.linspace(0,len(x)-1,tmp_deg_wvs,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
        Npix = spectrum.shape[-1]

        wvs_min = np.arange(wvs0[0]-N_dwv*dwv,wvs0[0]+N_dwv*dwv,dwv)#-130*dwv/2
        wvs_mid = np.arange(wvs0[Npix//3]-N_dwv*dwv,wvs0[Npix//3]+N_dwv*dwv,dwv)#-130*dwv/2
        wvs_mid2 = np.arange(wvs0[2*Npix//3]-N_dwv*dwv,wvs0[2*Npix//3]+N_dwv*dwv,dwv)#-130*dwv/2
        wvs_max = np.arange(wvs0[-1]-N_dwv*dwv,wvs0[-1]+N_dwv*dwv,dwv)#-125*dwv/2
        nloglike_arr = np.zeros((np.size(wvs_min),np.size(wvs_mid),np.size(wvs_mid2),np.size(wvs_max)))
        for k,wv_min in enumerate(wvs_min):
            for l, wv_mid in enumerate(wvs_mid):
                for l2, wv_mid2 in enumerate(wvs_mid2):
                    for m, wv_max in enumerate(wvs_max):
                        nloglike_arr[k,l,l2,m] = wavcal_nloglike([wv_min,wv_mid,wv_mid2,wv_max,1000,25], x, spectrum, spec_err, star_func, telluric_wvs, telluric_interpgrid, tmp_deg_wvs, blaze_chunks,False,rv)
        argmin2d = np.unravel_index(np.argmin(nloglike_arr),nloglike_arr.shape)

        spl = InterpolatedUnivariateSpline(x_knots, [wvs_min[argmin2d[0]], wvs_mid[argmin2d[1]], wvs_mid2[argmin2d[2]],
                                               wvs_max[argmin2d[3]]], k=3, ext=0)
        wvs00 = spl(x)
    else:
        wvs00 = wvs0

    ## Real initialization
    x_knots = x[np.linspace(0, len(x) - 1, N_nodes_wvs, endpoint=True).astype(np.int)]
    if fitsrv:
        paras0 = np.array(wvs00[x_knots].tolist() + [rv,1000,25])
    else:
        paras0 = np.array(wvs00[x_knots].tolist() + [1000,25])
    simplex_init_steps =  np.ones(np.size(paras0))
    simplex_init_steps[0:N_nodes_wvs] = dwv/4
    if fitsrv:
        simplex_init_steps[N_nodes_wvs] = 1
        simplex_init_steps[N_nodes_wvs+1] = 200
        simplex_init_steps[N_nodes_wvs+2] = 10
    else:
        simplex_init_steps[N_nodes_wvs] = 200
        simplex_init_steps[N_nodes_wvs+1] = 10
    initial_simplex = np.concatenate([paras0[None,:],paras0[None,:] + np.diag(simplex_init_steps)],axis=0)
    res = minimize(lambda paras: wavcal_nloglike(paras, x,spectrum,spec_err,star_func,telluric_wvs,telluric_interpgrid,N_nodes_wvs,blaze_chunks,fitsrv,rv), paras0, method="nelder-mead",
                           options={"xatol": 1e-10, "maxiter": 1e5,"initial_simplex":initial_simplex,"disp":False})
    out = res.x

    spl = InterpolatedUnivariateSpline(x_knots, out[0:N_nodes_wvs], k=3, ext=0)
    return spl(x),out,x_knots


def fit_wavecal_fib(init_wvs,combined_spec,combined_err,star_func,star_rv, telluric_wvs, telluric_interpgrid,
                    N_nodes_wvs=5, blaze_chunks=5,init_grid_search=False,init_grid_dwv=3e-4,mypool = None):
    """
    Args:

    Returns:
        out:
    """

    combined_spec = edges2nans(combined_spec)
    combined_err = edges2nans(combined_err)

    combined_spec_clip = np.zeros(combined_err.shape)
    for l in range(combined_spec.shape[0]):
        combined_spec_clip[l,:] = np.clip(combined_err[l,:], 0.1 * np.nanmedian(combined_err[l,:]), np.inf)

    new_wvs_arr = np.zeros(init_wvs.shape)
    x = np.arange(0,2048)
    fitsrv = False
    if fitsrv:
        out_paras = np.zeros((combined_spec.shape[0],N_nodes_wvs+3))
    else:
        out_paras = np.zeros((combined_spec.shape[0],N_nodes_wvs+2))

    if mypool is None: # jump
        for l in range(combined_spec.shape[0]):
        # for l in [6]:
            spectrum = combined_spec[l,:]
            spec_err = combined_spec_clip[l,:]

            wvs0 = init_wvs[l,:]

            new_wvs, out, x_knots = _fit_wavecal((x, wvs0, spectrum, spec_err, star_func, telluric_wvs,telluric_interpgrid,
                                                  N_nodes_wvs, blaze_chunks, fitsrv,star_rv,init_grid_search,init_grid_dwv))
            new_wvs_arr[l,:] = new_wvs
            out_paras[l,:] = out

        #     print(out)
        #     import matplotlib.pyplot as plt
        #     plt.figure(l+1)
        #     plt.plot(x, spectrum, label="data")
        #     plt.fill_between(x, spectrum - spec_err, spectrum + spec_err, label="data err", alpha=0.5)
        #     plt.plot(x, wavcal_model(out, x, spectrum, spec_err, star_func, telluric_wvs,telluric_interpgrid,
        #                                   N_nodes_wvs, blaze_chunks, fitsrv, star_rv), label="model")
        #     plt.legend()
        # plt.show()
        # exit()
    else:
        outputs_list = mypool.map(_fit_wavecal, zip(itertools.repeat(x),
                                                    init_wvs, combined_spec, combined_err,
                                                    itertools.repeat(star_func),
                                                    itertools.repeat(telluric_wvs),
                                                    itertools.repeat(telluric_interpgrid),
                                                    itertools.repeat(N_nodes_wvs),
                                                    itertools.repeat(blaze_chunks),
                                                    itertools.repeat(fitsrv),
                                                    itertools.repeat(star_rv),
                                                    itertools.repeat(init_grid_search),
                                                    itertools.repeat(init_grid_dwv)))

        for l, out in enumerate(outputs_list):
            new_wvs, out, x_knots = out
            new_wvs_arr[l, :] = new_wvs
            out_paras[l,:] = out

    model = np.zeros(combined_spec.shape)
    for l in range(combined_spec.shape[0]):
        spectrum = combined_spec[l,:]
        spec_err = copy(combined_spec_clip[l,:])
        model[l,:] = wavcal_model(out_paras[l,:], x, spectrum, spec_err, star_func, telluric_wvs,telluric_interpgrid,
                                      N_nodes_wvs, blaze_chunks, fitsrv, star_rv)

    return new_wvs_arr,model,out_paras