import numpy as np
from copy import copy
import astropy.io.fits as fits
from scipy.interpolate import interp1d
import itertools
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
import kpicdrp.utils as utils
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import loggamma
import matplotlib.pyplot as plt
import warnings
from PyAstronomy import pyasl
import multiprocessing as mp

import scipy.ndimage as ndi
from scipy.optimize import minimize

def open_psg_allmol(filename,l0,l1):
	"""
	Open psg model for all molecules
	returns wavelength, h2o, co2, ch4, co for l0-l1 range specified

	no o3 here .. make this more flexible
	--------
	"""
	f = fits.getdata(filename)

	x = f['Wave/freq']

	h2o = f['H2O']
	co2 = f['CO2']
	ch4 = f['CH4']
	co  = f['CO']
	o3  = f['O3'] # O3 messes up in PSG at lam<550nm and high resolution bc computationally expensive, so don't use it if l1<550
	n2o = f['N2O']
	o2  = f['O2']
	# also dont use rayleigh scattering, it fails at NIR wavelengths. absorbs into a continuum fit anyhow

	idelete = np.where(np.diff(x) < .0001)[0]  # delete non unique points - though I fixed it in code but seems to pop up still at very high resolutions
	x, h2o, co2, ch4, co, o3, n2o, o2= np.delete(x,idelete),np.delete(h2o,idelete), np.delete(co2,idelete),np.delete(ch4,idelete),np.delete(co,idelete),np.delete(o3,idelete),np.delete(n2o,idelete),np.delete(o2,idelete)

	isub = np.where((x > l0) & (x < l1))[0]
	return x[isub], (h2o[isub], co2[isub], ch4[isub], co[isub], o3[isub], n2o[isub], o2[isub])


def scale_psg(psg_tuple, airmass, pwv):
	"""
	psg_tuple : (tuple) of loaded psg spectral components from "open_psg_allmol" fxn
	airmass: (float) airmass of final spectrum applied to all molecules spectra
	pwv: (float) extra scaling for h2o spectrum to account for changes in the precipitable water vapor
	"""
	h2o, co2, ch4, co, o3, n2o, o2 = psg_tuple

	model = h2o**(airmass + pwv) * (co2 * ch4 * co * o3 * n2o * o2)**airmass # do the scalings

	return model


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
            atran_spec_conv = utils.convolve_spectrum_line_width(atran_wvs,atran_spec,atran_line_widths,mypool=mypool)
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

def wavcal_model(paras, x,spectrum,spec_err, star_func,telluric_wvs,telluric_interpgrid,N_nodes_wvs,blaze_chunks,fitsrv,rv,fringing):
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
    if fringing:
        F = paras[-1]
        delta = (2*np.pi)/wvs*paras[-2]
        tmp *= 1/(1+F*np.sin(delta/2)**2)
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

def wavcal_nloglike(paras, x,spectrum,spec_err, star_func,telluric_wvs,telluric_interpgrid,N_nodes_wvs,blaze_chunks,fitsrv,rv,fringing):
    m = wavcal_model(paras, x,spectrum,spec_err, star_func,telluric_wvs,telluric_interpgrid,N_nodes_wvs,blaze_chunks,fitsrv,rv,fringing)
    if m is not None:
        nloglike = np.nansum((spectrum - m) ** 2 / (spec_err) ** 2)
        return 1 / 2. * nloglike
    else:
        return np.inf

def _fit_wavecal(paras):
    x,wvs0,spectrum,spec_err, star_func,telluric_wvs,telluric_interpgrid,N_nodes_wvs,blaze_chunks,fitsrv,rv,\
        init_grid_search,init_grid_dwv,fringing = paras
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
                        nloglike_arr[k,l,l2,m] = wavcal_nloglike([wv_min,wv_mid,wv_mid2,wv_max,1000,25], x,
                                                                 spectrum, spec_err, star_func, telluric_wvs,
                                                                 telluric_interpgrid, tmp_deg_wvs, blaze_chunks,
                                                                 False,rv,False)
        argmin2d = np.unravel_index(np.argmin(nloglike_arr),nloglike_arr.shape)

        spl = InterpolatedUnivariateSpline(x_knots, [wvs_min[argmin2d[0]], wvs_mid[argmin2d[1]], wvs_mid2[argmin2d[2]],
                                               wvs_max[argmin2d[3]]], k=3, ext=0)
        wvs00 = spl(x)
    else:
        wvs00 = wvs0

    ## Real initialization
    x_knots = x[np.linspace(0, len(x) - 1, N_nodes_wvs, endpoint=True).astype(np.int)]
    paras0 = wvs00[x_knots].tolist()
    simplex_init_steps = [dwv/4,]*len(paras0)
    if fitsrv:
        paras0 = paras0 + [rv,]
        simplex_init_steps = simplex_init_steps + [1,]
    paras0 = paras0 + [1000, 25]
    simplex_init_steps = simplex_init_steps + [200,10]
    if fringing:
        paras0 = paras0 + [10852.852852852853,0.06606606606606608]
        simplex_init_steps = simplex_init_steps + [10,0.001]

    paras0 = np.array(paras0)
    initial_simplex = np.concatenate([paras0[None,:],paras0[None,:] + np.diag(simplex_init_steps)],axis=0)

    res = minimize(lambda paras: wavcal_nloglike(paras, x,spectrum,spec_err,star_func,telluric_wvs,telluric_interpgrid,N_nodes_wvs,blaze_chunks,fitsrv,rv,fringing), paras0, method="nelder-mead",
                           options={"xatol": 1e-6, "maxiter": 1e5,"initial_simplex":initial_simplex,"disp":False})
    out = res.x

    spl = InterpolatedUnivariateSpline(x_knots, out[0:N_nodes_wvs], k=3, ext=0)
    return spl(x),out,x_knots

def fit_wavecal_fib(init_wvs,combined_spec,combined_err,star_func,star_rv, telluric_wvs, telluric_interpgrid,
                    N_nodes_wvs=5, blaze_chunks=5,init_grid_search=False,init_grid_dwv=3e-4,fringing = False,mypool = None):
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
    out_paras = np.zeros((combined_spec.shape[0],N_nodes_wvs+1*fitsrv+2*fringing+2))

    if mypool is None: # jump
        for l in range(combined_spec.shape[0]):
            spectrum = combined_spec[l,:]
            spec_err = combined_spec_clip[l,:]

            wvs0 = init_wvs[l,:]

            new_wvs, out, x_knots = _fit_wavecal((x, wvs0, spectrum, spec_err, star_func, telluric_wvs,telluric_interpgrid,
                                                  N_nodes_wvs, blaze_chunks, fitsrv,star_rv,init_grid_search,init_grid_dwv,fringing))
            new_wvs_arr[l,:] = new_wvs
            out_paras[l,:] = out
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
                                                    itertools.repeat(init_grid_dwv),
                                                    itertools.repeat(fringing)))

        for l, out in enumerate(outputs_list):
            new_wvs, out, x_knots = out
            new_wvs_arr[l, :] = new_wvs
            out_paras[l,:] = out

    model = np.zeros(combined_spec.shape)
    for l in range(combined_spec.shape[0]):
        spectrum = combined_spec[l,:]
        spec_err = copy(combined_spec_clip[l,:])
        model[l,:] = wavcal_model(out_paras[l,:], x, spectrum, spec_err, star_func, telluric_wvs,telluric_interpgrid,
                                      N_nodes_wvs, blaze_chunks, fitsrv, star_rv,fringing)

    return new_wvs_arr,model,out_paras


def psg_wavcal_model(paras, x,spectrum,spec_err, star_func,telluric_wvs,psg_tuple,N_nodes_wvs,blaze_chunks,fitsrv,rv,fringing):
    c_kms = 299792.458
    wvs_coefs = paras[0:N_nodes_wvs]
    sig = 1
    if fitsrv:
        star_rv = paras[N_nodes_wvs]
        pwv = paras[N_nodes_wvs+1]
        airmass = paras[N_nodes_wvs+2]
    else:
        star_rv = rv
        pwv = paras[N_nodes_wvs]
        airmass = paras[N_nodes_wvs+1]

    if airmass <=0:
        return None
    if pwv <=0:
        return None

    tel_func = interp1d(telluric_wvs,scale_psg(psg_tuple, airmass, pwv),bounds_error=False,fill_value=0)

    x_knots = x[np.linspace(0,len(x)-1,N_nodes_wvs,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
    spl = InterpolatedUnivariateSpline(x_knots,wvs_coefs,k=np.min([3,N_nodes_wvs-1]),ext=0)
    wvs= spl(x)

    tmp = star_func(wvs*(1-star_rv/c_kms))*tel_func(wvs)
    if fringing:
        F = paras[-1]
        delta = (2*np.pi)/wvs*paras[-2]
        tmp *= 1/(1+F*np.sin(delta/2)**2)
    M0_mn = np.nanmean(tmp)
    tmp /= M0_mn

    x_knots = x[np.linspace(0,len(x)-1,blaze_chunks+1,endpoint=True).astype(np.int)]
    M = utils.get_spline_model(x_knots,x,spline_degree=np.min([blaze_chunks,3]))
    M = tmp[:,None]*M

    if 0:
        deg_off = 4
        x_knots = x[np.linspace(0,len(x)-1,deg_off+1,endpoint=True).astype(np.int)]
        Moff = utils.get_spline_model(x_knots,x,spline_degree=3)
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

def psg_wavcal_nloglike(paras, x,spectrum,spec_err, star_func,telluric_wvs,psg_tuple,N_nodes_wvs,blaze_chunks,fitsrv,rv,fringing):
    m = psg_wavcal_model(paras, x,spectrum,spec_err, star_func,telluric_wvs,psg_tuple,N_nodes_wvs,blaze_chunks,fitsrv,rv,fringing)
    if m is not None:
        nloglike = np.nansum((spectrum - m) ** 2 / (spec_err) ** 2)
        return 1 / 2. * nloglike
    else:
        return np.inf

def _fit_psg_wavecal(paras):
    x,wvs0,spectrum,spec_err, star_func,telluric_wvs,psg_tuple,N_nodes_wvs,blaze_chunks,fitsrv,rv,\
        init_grid_search,init_grid_dwv,fringing = paras
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
                        nloglike_arr[k,l,l2,m] = psg_wavcal_nloglike([wv_min,wv_mid,wv_mid2,wv_max,1,1.5], x,
                                                                 spectrum, spec_err, star_func, telluric_wvs,
                                                                 psg_tuple, tmp_deg_wvs, blaze_chunks,
                                                                 False,rv,False)
        argmin2d = np.unravel_index(np.argmin(nloglike_arr),nloglike_arr.shape)

        spl = InterpolatedUnivariateSpline(x_knots, [wvs_min[argmin2d[0]], wvs_mid[argmin2d[1]], wvs_mid2[argmin2d[2]],
                                               wvs_max[argmin2d[3]]], k=np.min([3,N_nodes_wvs-1]), ext=0)
        wvs00 = spl(x)
    else:
        wvs00 = wvs0

    ## Real initialization
    x_knots = x[np.linspace(0, len(x) - 1, N_nodes_wvs, endpoint=True).astype(np.int)]
    paras0 = wvs00[x_knots].tolist()
    simplex_init_steps = [dwv/4,]*len(paras0)
    if fitsrv:
        paras0 = paras0 + [rv,]
        simplex_init_steps = simplex_init_steps + [1,]
    paras0 = paras0 + [1,1.5]
    simplex_init_steps = simplex_init_steps + [0.25,0.25]
    if fringing:
        paras0 = paras0 + [10852.852852852853,0.06606606606606608]
        simplex_init_steps = simplex_init_steps + [10,0.001]

    paras0 = np.array(paras0)
    initial_simplex = np.concatenate([paras0[None,:],paras0[None,:] + np.diag(simplex_init_steps)],axis=0)

    # res = minimize(lambda paras: psg_wavcal_nloglike(paras, x,spectrum,spec_err,star_func,telluric_wvs,psg_tuple,N_nodes_wvs,blaze_chunks,fitsrv,rv,fringing), paras0, method="nelder-mead",
    #                        options={"xatol": 1e-6, "maxiter": 1e5,"initial_simplex":initial_simplex,"disp":False})

    # updated June 3, 2022 from JB's suggestion
    res = minimize(lambda paras: psg_wavcal_nloglike(paras, x,spectrum,spec_err,star_func,telluric_wvs,psg_tuple,N_nodes_wvs,blaze_chunks,fitsrv,rv,fringing), paras0, method="nelder-mead",
                           options={"xatol":np.inf,"fatol":0.1,"maxiter": 5e3,"initial_simplex":initial_simplex,"disp":False})
    out = res.x

    spl = InterpolatedUnivariateSpline(x_knots, out[0:N_nodes_wvs], k=3, ext=0)
    return spl(x),out,x_knots

def fit_psg_wavecal_fib(init_wvs,combined_spec,combined_err,star_func,star_rv, telluric_wvs, psg_tuple,
                    N_nodes_wvs=5, blaze_chunks=5,init_grid_search=False,init_grid_dwv=3e-4,fringing = False,mypool = None):
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
    out_paras = np.zeros((combined_spec.shape[0],N_nodes_wvs+1*fitsrv+2*fringing+2))

    if mypool is None: # jump
        for l in range(combined_spec.shape[0]):
            spectrum = combined_spec[l,:]
            spec_err = combined_spec_clip[l,:]

            wvs0 = init_wvs[l,:]

            new_wvs, out, x_knots = _fit_psg_wavecal((x, wvs0, spectrum, spec_err, star_func, telluric_wvs,psg_tuple,
                                                  N_nodes_wvs, blaze_chunks, fitsrv,star_rv,init_grid_search,init_grid_dwv,fringing))
            new_wvs_arr[l,:] = new_wvs
            out_paras[l,:] = out
    else:
        outputs_list = mypool.map(_fit_psg_wavecal, zip(itertools.repeat(x),
                                                    init_wvs, combined_spec, combined_err,
                                                    itertools.repeat(star_func),
                                                    itertools.repeat(telluric_wvs),
                                                    itertools.repeat(psg_tuple),
                                                    itertools.repeat(N_nodes_wvs),
                                                    itertools.repeat(blaze_chunks),
                                                    itertools.repeat(fitsrv),
                                                    itertools.repeat(star_rv),
                                                    itertools.repeat(init_grid_search),
                                                    itertools.repeat(init_grid_dwv),
                                                    itertools.repeat(fringing)))

        for l, out in enumerate(outputs_list):
            new_wvs, out, x_knots = out
            new_wvs_arr[l, :] = new_wvs
            out_paras[l,:] = out

    model = np.zeros(combined_spec.shape)
    for l in range(combined_spec.shape[0]):
        spectrum = combined_spec[l,:]
        spec_err = copy(combined_spec_clip[l,:])
        model[l,:] = psg_wavcal_model(out_paras[l,:], x, spectrum, spec_err, star_func, telluric_wvs,psg_tuple,
                                      N_nodes_wvs, blaze_chunks, fitsrv, star_rv,fringing)

    return new_wvs_arr,model,out_paras



def fitfm(nonlin_paras, fm_func, fm_paras,computeH0 = True):
    """
    Fit a forard model to data returning probabilities and best fit linear parameters.

    Args:
        nonlin_paras: [p1,p2,...] List of non-linear parameters such as rv, y, x. The meaning and number of non-linear
            parameters depends on the forward model defined.
        fm_func: A forward model function. See breads.fm.template.template() for an example.
        fm_paras: Additional parameters for fm_func (other than non-linear parameters)
        computeH0: If true (default), compute the probability of the model removing the first element of the linear
            model; See second ouput log_prob_H0. This can be used to compute the Bayes factor for a fixed set of
            non-linear parameters

    Returns:
        log_prob: Probability of the model marginalized over linear parameters.
        log_prob_H0: Probability of the model without the planet marginalized over linear parameters.
        rchi2: noise scaling factor
        linparas: Best fit linear parameters
        linparas_err: Uncertainties of best fit linear parameters
    """
    d,M,s = fm_func(nonlin_paras,**fm_paras)
    N_linpara = M.shape[1]
    if N_linpara == 1:
        computeH0 = False

    validpara = np.where(np.nansum(M,axis=0)!=0)
    M = M[:,validpara[0]]

    d = d / s
    M = M / s[:, None]

    N_data = np.size(d)
    linparas = np.ones(N_linpara)+np.nan
    linparas_err = np.ones(N_linpara)+np.nan
    if N_data == 0 or 0 not in validpara[0]:
        log_prob = -np.inf
        log_prob_H0 = -np.inf
        rchi2 = np.inf
    else:
        logdet_Sigma = np.sum(2 * np.log(s))
        paras = lsq_linear(M, d).x

        m = np.dot(M, paras)
        r = d  - m
        chi2 = np.nansum(r**2)
        rchi2 = chi2 / N_data

        # plt.figure()
        # for col in M.T:
        #     plt.plot(col / np.nanmean(col))
        # plt.show()


        covphi = rchi2 * np.linalg.inv(np.dot(M.T, M))
        slogdet_icovphi0 = np.linalg.slogdet(np.dot(M.T, M))

        log_prob = -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0[1] - (N_data - N_linpara + 2 - 1) / 2 * np.log(chi2) + \
                    loggamma((N_data - N_linpara + 2 - 1) / 2) + (N_linpara - N_data) / 2 * np.log(2 * np.pi)
        paras_err = np.sqrt(np.diag(covphi))

        if computeH0:
            paras_H0 = lsq_linear(M[:,1::], d).x
            m_H0 = np.dot(M[:,1::] , paras_H0)
            r_H0 = d  - m_H0
            chi2_H0 = np.nansum(r_H0**2)
            slogdet_icovphi0_H0 = np.linalg.slogdet(np.dot(M[:,1::].T, M[:,1::]))
            #todo check the maths when N_linpara is different from M.shape[1]. E.g. at the edge of the FOV
            log_prob_H0 = -0.5*logdet_Sigma - 0.5*slogdet_icovphi0_H0[1] - (N_data-1+N_linpara-1-1)/2*np.log(chi2_H0)+ \
                          loggamma((N_data-1+(N_linpara-1)-1)/2)+((N_linpara-1)-N_data)/2*np.log(2*np.pi)
        else:
            log_prob_H0 = np.nan

        linparas[validpara] = paras
        linparas_err[validpara] = paras_err

        # import matplotlib.pyplot as plt
        # print(log_prob, log_prob_H0, rchi2)
        # print(linparas)
        # print(linparas_err)
        # plt.plot(d,label="d")
        # plt.plot(m,label="m")
        # plt.plot(r,label="r")
        # plt.legend()
        # plt.show()


    return log_prob, log_prob_H0, rchi2, linparas, linparas_err

def log_prob(nonlin_paras, fm_func, fm_paras,nonlin_lnprior_func=None):
    """
    Wrapper to fit_fm() but only returns the log probability marginalized over the linear parameters.

    Args:
        nonlin_paras: [p1,p2,...] List of non-linear parameters such as rv, y, x. The meaning and number of non-linear
            parameters depends on the forward model defined.
        fm_func: A forward model function. See breads.fm.template.template() for an example.
        fm_paras: Additional parameters for fm_func (other than non-linear parameters)
        computeH0: If true (default), compute the probability of the model removing the first element of the linear
            model; See second ouput log_prob_H0. This can be used to compute the Bayes factor for a fixed set of
            non-linear parameters

    Returns:
        log_prob: Probability of the model marginalized over linear parameters.
    """
    if nonlin_lnprior_func is not None:
        prior = nonlin_lnprior_func(nonlin_paras)
    else:
        prior = 0
    try:
        lnprob = fitfm(nonlin_paras, fm_func, fm_paras,computeH0=False)[0]+prior
    except:
        lnprob =  -np.inf
    # print(lnprob)
    return lnprob


def nlog_prob(nonlin_paras, fm_func, fm_paras,nonlin_lnprior_func=None):
    """
   Returns the negative of the log_prob() for minimization routines.

    Args:
        nonlin_paras: [p1,p2,...] List of non-linear parameters such as rv, y, x. The meaning and number of non-linear
            parameters depends on the forward model defined.
        fm_func: A forward model function. See breads.fm.template.template() for an example.
        fm_paras: Additional parameters for fm_func (other than non-linear parameters)
        computeH0: If true (default), compute the probability of the model removing the first element of the linear
            model; See second ouput log_prob_H0. This can be used to compute the Bayes factor for a fixed set of
            non-linear parameters

    Returns:
        log_prob: Probability of the model marginalized over linear parameters.
    """
    return - log_prob(nonlin_paras, fm_func, fm_paras,nonlin_lnprior_func)


caf2_args = [1.03032805, 4.03776479e-3, 4.09367374, 5.60918206e-3, 5.54020715e-2, 1.264110033e3]
caf2_args = [[1.04834, -2.21666E-04, -6.73446E-06, 1.50138E-08, -2.77255E-11],
             [-3.32723E-03, 2.34683E-04, 6.55744E-06, -1.47028E-08, 2.75023E-11],
             [3.72693, 1.49844E-02, -1.47511E-04, 5.54293E-07, -7.17298E-10],
             [7.94375E-02, -2.20758E-04, 2.07862E-06, -9.60254E-09, 1.31401E-11],
             [0.258039, -2.12833E-03, 1.20393E-05, -3.06973E-08, 2.79793E-11],
             [34.0169, 6.26867E-02, -6.14541E-04, 2.31517E-06, -2.99638E-09]]


def sellmeir1(wvs, temp, K1, K2, K3, L1, L2, L3):
    """
    Compute index of refraction of a material given some coefficients

    Returns:
        n: index of refraction
    """
    wvs2 = wvs ** 2

    K1_total = np.sum([K1[i] * temp ** i for i in range(len(K1))])
    K2_total = np.sum([K2[i] * temp ** i for i in range(len(K1))])
    K3_total = np.sum([K3[i] * temp ** i for i in range(len(K1))])

    L1_total = np.sum([L1[i] * temp ** i for i in range(len(K1))])
    L2_total = np.sum([L2[i] * temp ** i for i in range(len(K1))])
    L3_total = np.sum([L3[i] * temp ** i for i in range(len(K1))])

    # print(K1_total, K2_total, K3_total, L1_total, L2_total, L3_total)

    arg1 = K1_total * wvs2 / (wvs2 - L1_total ** 2)
    arg2 = K2_total * wvs2 / (wvs2 - L2_total ** 2)
    arg3 = K3_total * wvs2 / (wvs2 - L3_total ** 2)

    n = np.sqrt(arg1 + arg2 + arg3 + 1)

    return n

def psg_wavcal_fm(nonlin_paras, spectrum,spec_err, line_width_func,stellar_model_wvs,stellar_model_grid,
                     telluric_wvs,psg_tuple,N_nodes_wvs,blaze_chunks,
                     simplewvsfit=True,wvs_init = None,baryrv=0,fix_parameters=None,quickinstbroadening=True, fixed_spec_func= None,
                     extra_outputs=False):
    out_newwvs = np.zeros(spectrum.shape)

    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters)==None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras
    c_kms = 299792.458
    if simplewvsfit:
        N_nodes_wvs = 2
    # else:
    #     N_nodes_wvs = None
    N_orders, nz = spectrum.shape

    wvs_allcoefs = _nonlin_paras[0:N_nodes_wvs*N_orders]
    if fixed_spec_func is None:
        star_rv = _nonlin_paras[N_nodes_wvs*N_orders]
        vsini = _nonlin_paras[N_nodes_wvs*N_orders+1]
        teff = _nonlin_paras[N_nodes_wvs*N_orders+2]
        logg = _nonlin_paras[N_nodes_wvs*N_orders+3]
        Z = _nonlin_paras[N_nodes_wvs*N_orders+4]
        airmass = _nonlin_paras[N_nodes_wvs*N_orders+5]
        pwv = _nonlin_paras[N_nodes_wvs*N_orders+6]
        fringing_Tmat = _nonlin_paras[N_nodes_wvs*N_orders+7]
        fringing_OPD = _nonlin_paras[N_nodes_wvs*N_orders+8]
        fringing_ghostampl = _nonlin_paras[N_nodes_wvs*N_orders+9]
        bad_paras = airmass <=0 or pwv <0
    else:
        bad_paras = False

    N_linpara = blaze_chunks+1

    where_finite = np.where(np.isfinite(np.ravel(spectrum)))
    x = np.arange(nz)

    badpixfraction = 0.75
    if np.size(where_finite[0]) <= (1-badpixfraction) * np.size(spectrum) or bad_paras:
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:

        M_list = []
        for orderid in range(N_orders):
            wvs_coefs = wvs_allcoefs[orderid * N_nodes_wvs:(orderid + 1) * N_nodes_wvs]
            if simplewvsfit:
                new_wvs = wvs_init[orderid, :]  + wvs_coefs[0] + (wvs_init[orderid, :]-np.mean(wvs_init[orderid, :])) * (wvs_coefs[1])
            else:
                x_knots = x[np.linspace(0, len(x) - 1, N_nodes_wvs, endpoint=True).astype(np.int)]  # np.array([wvs_stamp[wvid] for wvid in )
                spl = InterpolatedUnivariateSpline(x_knots, wvs_coefs, k=np.min([3, N_nodes_wvs - 1]), ext=0)
                new_wvs = spl(x)
            out_newwvs[orderid, :] = new_wvs

        if fixed_spec_func is None:
            reduc_factor = 40
            _telluric_wvs = telluric_wvs[0:(np.size(telluric_wvs) // reduc_factor) * reduc_factor]
            _telluric_wvs = np.mean(np.reshape(_telluric_wvs, ((np.size(_telluric_wvs) // reduc_factor), reduc_factor)), axis=1)
            telluric_spec = scale_psg(psg_tuple, airmass, pwv)
            telluric_spec = telluric_spec[0:(np.size(telluric_spec) // reduc_factor) * reduc_factor]
            telluric_spec = np.mean(np.reshape(telluric_spec, ((np.size(telluric_spec) // reduc_factor), reduc_factor)), axis=1)

            reduc_factor = 5
            stellar_model_spec = stellar_model_grid([teff, logg, Z])[0]
            stellar_model_spec_avg = np.array(pd.DataFrame(stellar_model_spec).rolling(window=reduc_factor,center=True).mean())[:, 0]
            stellar_model_wvs_avg = np.array(pd.DataFrame(stellar_model_wvs).rolling(window=reduc_factor,center=True).mean())[:, 0]
            stellar_model_spec_avg_func = interp1d(stellar_model_wvs_avg, stellar_model_spec_avg, bounds_error=False, fill_value=0)

            telluric_wvs_list = []
            conv_order_spec_list = []
            for orderid in range(N_orders):
                minwv = np.min(out_newwvs[orderid,:])
                maxwv = np.max(out_newwvs[orderid,:])
                if orderid == 0:
                    maxwv1 = np.nan
                else:
                    maxwv1 = np.max(out_newwvs[orderid-1,:])
                if orderid == N_orders-1:
                    minwv2 = np.nan
                else:
                    minwv2 = np.min(out_newwvs[orderid+1,:])
                min_bound = np.nanmax([(minwv-(maxwv-minwv)/1),(maxwv1+minwv)/2])
                max_bound = np.nanmin([(maxwv+(maxwv-minwv)/1),(maxwv+minwv2)/2])
                where_order_wvs = np.where((min_bound<_telluric_wvs)*(_telluric_wvs<max_bound))
                telluric_order_wvs = _telluric_wvs[where_order_wvs]
                telluric_order_spec = telluric_spec[where_order_wvs]
                stellar_order_spec = stellar_model_spec_avg_func(telluric_order_wvs* (1 - (star_rv-baryrv) / c_kms))
                if vsini != 0:
                    spinbroad_stellar_order_spec = pyasl.fastRotBroad(telluric_order_wvs, stellar_order_spec, 0.1, vsini)
                else:
                    spinbroad_stellar_order_spec = stellar_order_spec
                highres_order_spec = telluric_order_spec * spinbroad_stellar_order_spec


                #|S+A*S*exp(i*2pi*d/lambda_vac*n)|
                if fringing_ghostampl > 0:
                    n_mat = sellmeir1(telluric_order_wvs, fringing_Tmat, caf2_args[0], caf2_args[1], caf2_args[2], caf2_args[3], caf2_args[4], caf2_args[5])
                    # import matplotlib.pyplot as plt
                    # plt.plot(highres_order_spec,label="ori")
                    # highres_order_spec_fringing = highres_order_spec*np.abs(1+fringing_ghostampl*np.exp(2*np.pi*1j*n_mat*fringing_OPD/telluric_order_wvs))
                    phi = 2*np.pi*n_mat*fringing_OPD/telluric_order_wvs
                    highres_order_spec_fringing = highres_order_spec*(1+fringing_ghostampl**2+fringing_ghostampl*np.cos(phi))
                    highres_order_spec = highres_order_spec_fringing/np.nansum(highres_order_spec_fringing)*np.nansum(highres_order_spec)
                    # plt.plot(highres_order_spec_fringing,label="fringing")
                    # # plt.plot(highres_order_spec-highres_order_spec_fringing,label="difference")
                    # plt.legend()
                    # plt.figure(10)
                    # # plt.plot(telluric_order_wvs,np.abs(1+fringing_ghostampl*np.exp(2*np.pi*1j*n_mat*fringing_OPD/telluric_order_wvs)))
                    # # plt.plot(n_mat*fringing_OPD/telluric_order_wvs)
                    # plt.plot(np.abs(1+fringing_ghostampl*np.exp(2*np.pi*1j*n_mat*fringing_OPD/telluric_order_wvs)))
                    # plt.plot((1+fringing_ghostampl**2+fringing_ghostampl*np.cos(phi)))
                    # plt.show()


                if quickinstbroadening:
                    star_model_r = np.median(telluric_order_wvs) / np.median(telluric_order_wvs - np.roll(telluric_order_wvs, 1))
                    star_model_downsample = star_model_r / 35000 / (2 * np.sqrt(2 * np.log(2)))
                    conv_order_spec = ndi.gaussian_filter(highres_order_spec, star_model_downsample)
                else:
                    # broaden and create interpolation function for the Phoenix model
                    line_widths = np.array(pd.DataFrame(line_width_func(telluric_order_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                    conv_order_spec = utils.convolve_spectrum_line_width(telluric_order_wvs, highres_order_spec, line_widths,mypool=None)

                # print("downsample", star_model_downsample)
                # import matplotlib.pyplot as plt
                # plt.plot(conv_order_spec,label="jb")
                # plt.plot(conv_order_spec2,label="jason")
                # plt.legend()
                # plt.show()
                #
                # exit()
                conv_order_spec = conv_order_spec / np.nanmax(conv_order_spec)
                telluric_wvs_list.append(telluric_order_wvs)
                conv_order_spec_list.append(conv_order_spec)

            conv_order_spec_func = interp1d(np.concatenate(telluric_wvs_list,axis=0), np.concatenate(conv_order_spec_list,axis=0), bounds_error=False,fill_value=np.nan)
        else:
            conv_order_spec_func = fixed_spec_func

        x_knots = x[np.linspace(0, len(x) - 1, blaze_chunks + 1, endpoint=True).astype(np.int)]
        M0 = utils.get_spline_model(x_knots, x, spline_degree=np.min([blaze_chunks, 3]))
        for orderid in range(N_orders):
            new_wvs = out_newwvs[orderid,:]
            tmp = conv_order_spec_func(new_wvs)
            tmp = (tmp / np.nanmean(tmp))
            # if extra_outputs:
            #     # plt.subplot(2,1,1)
            #     plt.plot(tmp,label="1")
            #     # plt.subplot(2,1,2)
            #     # plt.plot(np.concatenate(telluric_wvs_list,axis=0), np.concatenate(conv_order_spec_list,axis=0))
            tmp_M = tmp[:, None] * M0
            M_extended = np.zeros((N_orders*nz,M0.shape[1]))
            M_extended[orderid*nz:(orderid+1)*nz,:] = tmp_M
            M_list.append(M_extended)


        # combine planet model with speckle model
        M = np.concatenate(M_list, axis=1)
        # Get rid of bad pixels
        sr = np.ravel(spec_err)[where_finite]
        dr = np.ravel(spectrum)[where_finite]
        Mr = M[where_finite[0], :]

        if extra_outputs:
            return dr, Mr, sr,out_newwvs,conv_order_spec_func
        else:
            return dr, Mr, sr


# def get_new_wvs(nonlin_paras,wvs_init,spectrum_shape,simplefit=True,fix_parameters=None):
#     if fix_parameters is not None:
#         _nonlin_paras = np.array(fix_parameters)
#         _nonlin_paras[np.where(np.array(fix_parameters) == None)] = nonlin_paras
#     else:
#         _nonlin_paras = nonlin_paras
#     if simplefit:
#         N_nodes_wvs = 2
#     else:
#         N_nodes_wvs = None
#     N_orders, nz = spectrum_shape
#     wvs_allcoefs = _nonlin_paras[0:N_nodes_wvs*N_orders]
#     new_wvs = np.zeros(wvs_init.shape)
#     x = np.arange(nz)
#     for orderid in range(N_orders):
#         wvs_coefs = wvs_allcoefs[orderid*N_nodes_wvs:(orderid+1)*N_nodes_wvs]
#         if simplefit:
#             new_wvs[orderid,:] = wvs_init[orderid,:]*(1+wvs_coefs[1])+wvs_coefs[0]
#         else:
#             x_knots = x[np.linspace(0,len(x)-1,N_nodes_wvs,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
#             spl = InterpolatedUnivariateSpline(x_knots,wvs_coefs,k=np.min([3,N_nodes_wvs-1]),ext=0)
#             new_wvs[orderid,:] = spl(x)
#
#     return new_wvs


def process_chunk(args):
    """
    Process for search_planet()
    """
    nonlin_paras_list, fm_func, fm_paras = args

    for k, nonlin_paras in enumerate(zip(*nonlin_paras_list)):
        try:
        # if 1:
            log_prob,log_prob_H0,rchi2,linparas,linparas_err = fitfm(nonlin_paras,fm_func,fm_paras)
            N_linpara = np.size(linparas)
            if k == 0:
                out_chunk = np.zeros((np.size(nonlin_paras_list[0]),1+1+1+2*N_linpara))+np.nan
            out_chunk[k,0] = log_prob
            out_chunk[k,1] = log_prob_H0
            out_chunk[k,2] = rchi2
            out_chunk[k,3:(N_linpara+3)] = linparas
            out_chunk[k,(N_linpara+3):(2*N_linpara+3)] = linparas_err
        except Exception as e:
            print(e)
            print(nonlin_paras)
    return out_chunk


def grid_search(para_vecs,fm_func,fm_paras,numthreads=None):
    """
    Planet detection, CCF, or grid search routine.
    It fits for the non linear parameters of a forward model over a user-specified grid of values while marginalizing
    over the linear parameters. For a planet detection or CCF routine, choose a forward model (fm_func) and provide a
    grid of x,y, and RV values.

    SNR (detection maps or CCFs) can be computed as:
    N_linpara = (out.shape[-1]-2)//2
    snr = out[...,3]/out[...,3+N_linpara]

    The natural logarithm of the Bayes factor can be computed as
    bayes_factor = out[...,0] - out[...,1]

    Args:
        para_vecs: [vec1,vec2,...] List of 1d arrays defining the sampling of the grid of non-linear parameters such as
            rv, y, x. The meaning and number of non-linear parameters depends on the forward model defined.
        fm_func: A forward model function. See breads.fm.template.template() for an example.
        fm_paras: Additional parameters for fm_func (other than non-linear parameters)
        numthreads: Number of processes to be used in parallelization. Non parallization if defined as None (default).

    Returns:
        Out: An array of dimension (Nnl_1,Nnl_2,....,3+Nl*2) containing the marginalized probabilities, the noise
            scaling factor, and best fit linear parameters with associated uncertainties calculated over the grid of
            non-linear parameters. (Nnl_1,Nnl_2,..) is the shape of the non linear parameter grid with Nnl_1 the size of
            para_vecs[1] and so on. Nl is the number of linear parameters in the forward model. The last dimension
            is defined as follow:
                Out[:,...,0]: Probability of the model marginalized over linear parameters.
                Out[:,...,1]: Probability of the model without the planet marginalized over linear parameters.
                Out[:,...,2]: noise scaling factor
                Out[:,...,3:3+Nl]: Best fit linear parameters
                Out[:,...,3+Nl:3+2*Nl]: Uncertainties of best fit linear parameters

    """
    para_grids = [np.ravel(pgrid) for pgrid in np.meshgrid(*para_vecs,indexing="ij")]

    if numthreads is None:
        _out = process_chunk((para_grids,fm_func,fm_paras))
        out_shape = [np.size(v) for v in para_vecs]+[_out.shape[-1],]
        out = np.reshape(_out,out_shape)
    else:
        mypool = mp.Pool(processes=numthreads)
        chunk_size = np.max([1,np.size(para_grids[0])//(3*numthreads)])
        N_chunks = np.size(para_grids[0])//chunk_size
        nonlin_paras_lists = []
        indices_lists = []
        for k in range(N_chunks-1):
            nonlin_paras_lists.append([pgrid[(k*chunk_size):((k+1)*chunk_size)] for pgrid in para_grids])
            indices_lists.append(np.arange((k*chunk_size),((k+1)*chunk_size)))
        nonlin_paras_lists.append([pgrid[((N_chunks-1)*chunk_size):np.size(para_grids[0])] for pgrid in para_grids])
        indices_lists.append(np.arange(((N_chunks-1)*chunk_size),np.size(para_grids[0])))

        output_lists = mypool.map(process_chunk, zip(nonlin_paras_lists,
                                                     itertools.repeat(fm_func),
                                                     itertools.repeat(fm_paras)))

        for k,(indices, output_list) in enumerate(zip(indices_lists,output_lists)):
            if k ==0:
                out_shape = [np.size(v) for v in para_vecs]+[np.size(output_list[0]),]
                out = np.zeros(out_shape)
            for l,outvec in zip(indices,output_list):
                out[np.unravel_index(l,out_shape[0:len(para_vecs)])] = outvec

        mypool.close()
        mypool.join()
    return out

def optimize_wavcal(fm_func, fm_paras,paras0,nonlin_paras_mins,nonlin_paras_maxs,fix_parameters=None,simplex_init_steps=None,fatol=0.01,showplot=False,disp=True):
    if simplex_init_steps is None:
        simplex_init_steps = (np.array(nonlin_paras_maxs) - np.array(nonlin_paras_mins))/5

    if fix_parameters is not None:
        _nonlin_paras_mins = np.array(nonlin_paras_mins)[np.where(np.array(fix_parameters)==None)]
        _nonlin_paras_maxs = np.array(nonlin_paras_maxs)[np.where(np.array(fix_parameters)==None)]
        _paras0 = np.array(paras0)[np.where(np.array(fix_parameters)==None)]
        _simplex_init_steps = np.array(simplex_init_steps)[np.where(np.array(fix_parameters)==None)]
    else:
        _nonlin_paras_mins = nonlin_paras_mins
        _nonlin_paras_maxs = nonlin_paras_maxs
        _paras0 = paras0
        _simplex_init_steps = simplex_init_steps

    initial_simplex = np.concatenate([_paras0[None,:],_paras0[None,:] + np.diag(_simplex_init_steps)],axis=0)

    # def nonlin_lnprior_func(nonlin_paras):
    #     for p, _min, _max in zip(nonlin_paras, _nonlin_paras_mins, _nonlin_paras_maxs):
    #         if p > _max or p < _min:
    #             return -np.inf
    #     return 0
    bounds = [(bmin,bmax) for bmin,bmax in zip(_nonlin_paras_mins,_nonlin_paras_maxs)]
    nonlin_lnprior_func =None

    mini_out = minimize(nlog_prob, _paras0, args=(fm_func, fm_paras,nonlin_lnprior_func), method="nelder-mead",bounds=bounds,
                           options={"xatol":np.inf,"fatol":fatol,"maxiter": 5e3,"initial_simplex":initial_simplex,"disp":disp})
    d, M, s,new_wvs,model_func = fm_func(mini_out.x,extra_outputs=True, **fm_paras)

    out = {"new_wvs":new_wvs,"model_func":model_func,"mini_out":mini_out}

    validpara = np.where(np.sum(M, axis=0) != 0)
    where_finite = np.where(np.isfinite(np.ravel(fm_paras["spectrum"])))

    M = M[:, validpara[0]]
    d = d / s
    M = M / s[:, None]
    # from scipy.optimize import lsq_linear
    paras = lsq_linear(M, d).x
    m = np.dot(M, paras)

    N_order,nz = fm_paras["spectrum"].shape
    out_m = np.zeros(fm_paras["spectrum"].shape) + np.nan
    out_r = np.zeros(fm_paras["spectrum"].shape) + np.nan
    out_m.shape = (np.size(out_m),)
    out_m[:][where_finite] = m * s
    out_m.shape = fm_paras["spectrum"].shape
    out_r.shape = (np.size(out_m),)
    out_r[:][where_finite] = (d - m) * s
    out_r.shape = fm_paras["spectrum"].shape

    # blaze_chunks = fm_paras["blaze_chunks"]
    # if np.size(validpara[0]) != (blaze_chunks+1)*N_order:
    #     raise
    # x = np.arange(fm_paras["spectrum"].shape[1])
    # x_knots = x[np.linspace(0, len(x) - 1, blaze_chunks + 1, endpoint=True).astype(np.int)]
    # M0 = utils.get_spline_model(x_knots, x, spline_degree=np.min([blaze_chunks, 3]))
    # for orderid in range(fm_paras["spectrum"].shape[1]):


    out["best fit model"] = out_m
    out["residuals"] = out_r


    if showplot:
        plt.subplot(2, 1, 1)
        plt.plot(d, label="data")
        plt.plot(m, label="model")
        plt.plot(d - m, label="residuals")
        plt.legend()
        plt.subplot(2, 1, 2)
        for k in range(M.shape[-1]):
            plt.plot(M[:, k] / np.nanmax(M[:, k]), label="submodel {0}".format(k + 1))
        plt.legend()
        plt.show()

    return out