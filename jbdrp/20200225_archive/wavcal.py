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

def _task_convolve_spectrum_pixel_width(paras):
    indices,wvs,spectrum,pixel_widths = paras

    conv_spectrum = np.zeros(np.size(indices))
    dwvs = wvs[1::]-wvs[0:(np.size(wvs)-1)]
    dwvs = np.append(dwvs,dwvs[-1])
    for l,k in enumerate(indices):
        pwv = wvs[k]
        pix_width = pixel_widths[k]
        w = int(np.round(pix_width/dwvs[k]*2))
        stamp_spec = spectrum[np.max([0,k-w]):np.min([np.size(spectrum),k+w])]
        stamp_wvs = wvs[np.max([0,k-w]):np.min([np.size(wvs),k+w])]
        stamp_rel_wvs =  np.abs(stamp_wvs-pwv)
        hatkernel = np.array(stamp_rel_wvs<pix_width).astype(np.int)
        conv_spectrum[l] = np.sum(hatkernel*stamp_spec)/np.sum(hatkernel)
    return conv_spectrum

def convolve_spectrum_pixel_width(wvs,spectrum,pixel_widths,mypool=None):
    if mypool is None:
        return _task_convolve_spectrum_pixel_width((np.arange(np.size(spectrum)).astype(np.int),wvs,spectrum,pixel_widths))
    else:
        conv_spectrum = np.zeros(spectrum.shape)

        chunk_size=100
        N_chunks = np.size(spectrum)//chunk_size
        indices_list = []
        for k in range(N_chunks-1):
            indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int))
        indices_list.append(np.arange((N_chunks-1)*chunk_size,np.size(spectrum)).astype(np.int))
        outputs_list = mypool.map(_task_convolve_spectrum_pixel_width, zip(indices_list,
                                                               itertools.repeat(wvs),
                                                               itertools.repeat(spectrum),
                                                               itertools.repeat(pixel_widths)))
        for indices,out in zip(indices_list,outputs_list):
            conv_spectrum[indices] = out

        return conv_spectrum

def _task_convolve_spectrum_line_width(paras):
    indices,wvs,spectrum,line_widths = paras

    conv_spectrum = np.zeros(np.size(indices))
    dwvs = wvs[1::]-wvs[0:(np.size(wvs)-1)]
    dwvs = np.append(dwvs,dwvs[-1])
    for l,k in enumerate(indices):
        pwv = wvs[k]
        sig = line_widths[k]
        w = int(np.round(sig/dwvs[k]*10.))
        stamp_spec = spectrum[np.max([0,k-w]):np.min([np.size(spectrum),k+w])]
        stamp_wvs = wvs[np.max([0,k-w]):np.min([np.size(wvs),k+w])]
        stamp_dwvs = stamp_wvs[1::]-stamp_wvs[0:(np.size(stamp_spec)-1)]
        gausskernel = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(stamp_wvs-pwv)**2/sig**2)
        conv_spectrum[l] = np.sum(gausskernel[1::]*stamp_spec[1::]*stamp_dwvs)
    return conv_spectrum

def convolve_spectrum_line_width(wvs,spectrum,line_widths,mypool=None):
    if mypool is None:
        return _task_convolve_spectrum_line_width((np.arange(np.size(spectrum)).astype(np.int),wvs,spectrum,line_widths))
    else:
        conv_spectrum = np.zeros(spectrum.shape)

        chunk_size=100
        N_chunks = np.size(spectrum)//chunk_size
        indices_list = []
        for k in range(N_chunks-1):
            indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int))
        indices_list.append(np.arange((N_chunks-1)*chunk_size,np.size(spectrum)).astype(np.int))
        outputs_list = mypool.map(_task_convolve_spectrum_line_width, zip(indices_list,
                                                               itertools.repeat(wvs),
                                                               itertools.repeat(spectrum),
                                                               itertools.repeat(line_widths)))
        for indices,out in zip(indices_list,outputs_list):
            conv_spectrum[indices] = out

        return conv_spectrum


def wavcal_model_poly(paras, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv):
    c_kms = 299792.458
    wvs_coefs = paras[0:deg_wvs+1]
    sig = 1
    if fitstarrv is not None:
        star_rv = paras[deg_wvs+1]
        water = paras[deg_wvs+2]
    else:
        star_rv = 7.99 + 18.746#7.99+18.746
        water = paras[deg_wvs+1]

    if 0:
        wvs= np.polyval(wvs_coefs,x)
        # cont= np.polyval(cont_coefs,x)
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



def wavcal_nloglike_poly(paras, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv):
    # A, w, y0, B, rn, g = paras
    # sigs= (rn+g*np.abs(datacol))
    # nloglike = np.nansum((datacol-profile_model([A, w, y0, B],y))**2/sigs**2) + np.size(datacol)*np.log10(2*np.pi) + 2*np.sum(np.log10(sigs))
    # print(paras)

    # nloglike = np.nansum((spectrum - wavcal_model_poly(paras, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont)) ** 2 / (paras[-1]*spec_err) ** 2) + \
    #            np.nansum(np.log10(2 * np.pi * (paras[-1]*spec_err) ** 2))
    m = wavcal_model_poly(paras, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv)
    if m is not None:
        nloglike = np.nansum((spectrum - m) ** 2 / (spec_err) ** 2)
        return 1 / 2. * nloglike
    else:
        return np.inf

def _fit_wavcal_poly(paras):
    x,wvs0,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv = paras
    dwv = 0.3/2048
    if 0:
        if 0:
            offsets = np.arange(-100,100)
            ccf = np.zeros(offsets.shape)
            for k, offset in enumerate(offsets):
                wvs_coefs0 = np.polyfit(x+offset, wvs0, deg=deg_wvs)
                paras0 = wvs_coefs0.tolist()+[5000]
                m = wavcal_model_poly(paras0, x, spectrum, spec_err,instr_trans, phoenix_HR8799_func, atran_2d_func, deg_wvs, 1)
                ccf[k] = np.nansum(m*spectrum)
            #     if offset == 0:
            #         plt.plot(m,label="{0}".format(offset))
            #     if offset == -13:
            #         plt.plot(m,label="{0}".format(offset))

            wvs_coefs0 = np.polyfit(x + offsets[np.argmax(ccf)], wvs0, deg=deg_wvs)
        else:
            wvs_coefs0 = np.polyfit(x, wvs0, deg=deg_wvs)
            paras0 = wvs_coefs0.tolist()+[5000]
    else:
        if 1:
            # plt.plot()
            tmp_deg_wvs = 2
            N_dwv = 5
            wvs_min = np.arange(wvs0[0]-N_dwv*dwv,wvs0[0]+N_dwv*dwv,dwv)
            wvs_mid = np.arange(wvs0[2048//2]-N_dwv*dwv,wvs0[2048//2]+N_dwv*dwv,dwv)
            wvs_max = np.arange(wvs0[-1]-N_dwv*dwv,wvs0[-1]+N_dwv*dwv,dwv)
            nloglike_arr = np.zeros((np.size(wvs_min),np.size(wvs_mid),np.size(wvs_max)))
            for k,wv_min in enumerate(wvs_min):
                # print(k)
                for l, wv_mid in enumerate(wvs_mid):
                    for m, wv_max in enumerate(wvs_max):
                        nloglike_arr[k,l,m] = wavcal_nloglike_poly([wv_min,wv_mid,wv_max,5000], x, spectrum, spec_err,instr_trans, phoenix_HR8799_func, atran_2d_func, 2, deg_cont,None)
            argmin2d = np.unravel_index(np.argmin(nloglike_arr),nloglike_arr.shape)
            # print(argmin2d) #(99, 99)
            # plt.figure(1)
            # plt.imshow(nloglike_arr)
            # plt.clim([np.argmin(nloglike_arr),np.argmin(nloglike_arr)*1.1])
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

            x_knots = x[np.linspace(0, len(x) - 1, deg_wvs + 1, endpoint=True).astype(np.int)]  # np.array([wvs_stamp[wvid] for wvid in )
            # print(x_knots)
            # print(wvs2.shape)
            # exit()
            if fitstarrv is not None:
                paras0 = np.array(wvs2[x_knots].tolist() + [fitstarrv,8000])
            else:
                paras0 = np.array(wvs2[x_knots].tolist() + [8000])
            # print(wvs2[x_knots])
            # x_knots = x[np.linspace(0, len(x) - 1, deg_wvs + 1, endpoint=True).astype(np.int)]
            # spl = InterpolatedUnivariateSpline(x_knots, wvs2[x_knots],k=3, ext=0)
            # # plt.plot(x_knots, wvs2[x_knots],"x")
            # # plt.show()
            # wvs22 = spl(x)
            # plt.plot((wvs22-wvs2)/(0.3/2048))
            # plt.show()
            # paras0 = [wvs_min[argmin2d[0]],wvs_mid[argmin2d[1]],wvs_max[argmin2d[2]],5000]
            # plt.figure()
            # plt.plot(x,wavcal_model_poly(paras0, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,2,deg_cont),label="model")
            # plt.plot(x,spectrum,label="data")
            # plt.fill_between(x,spectrum-spec_err,spectrum+spec_err,label="data err",alpha=0.5)
            # plt.legend()
            # plt.show()
            # plt.show()
        # else:
        #     M2 = np.zeros((np.size(x),(deg_wvs+1)))
        #     x_knots = x[np.linspace(0,len(x)-1,deg_wvs+1,endpoint=True).astype(np.int)]#np.array([wvs_stamp[wvid] for wvid in )
        #     for polypower in range(deg_wvs):
        #         if polypower == 0:
        #             where_chunk = np.where((x_knots[polypower]<=x)*(x<=x_knots[polypower+1]))
        #         else:
        #             where_chunk = np.where((x_knots[polypower]<x)*(x<=x_knots[polypower+1]))
        #         M2[where_chunk[0],polypower] = 1-(x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
        #         M2[where_chunk[0],1+polypower] = (x[where_chunk]-x_knots[polypower])/(x_knots[polypower+1]-x_knots[polypower])
        #     p,chi2,rank,s = np.linalg.lstsq(M2,wvs2,rcond=None)
        #     paras0 = np.array(p.tolist()+[8000])
        #     wvs2 = np.dot(M2,p)
        #     # plt.plot(wvs0)
        #     # plt.plot(wvs2)
        #     # plt.show()

    # plt.plot(x,wavcal_model_poly(paras0, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont),label="model")
    # plt.plot(x,spectrum,label="data")
    # plt.fill_between(x,spectrum-spec_err,spectrum+spec_err,label="data err",alpha=0.5)
    # plt.legend()
    # plt.show()
    # plt.plot(spectrum)
    # plt.legend()
    # plt.figure(2)
    # plt.plot(np.arange(-100, 100), ccf)
    # plt.show()
    # cont_coefs0 = np.zeros((deg_cont+1,))
    # cont_coefs0[-1] = 1
    # paras0 = wvs_coefs0.tolist()+cont_coefs0.tolist()+[0,5000,1]
    # paras0 = wvs_coefs0.tolist()+cont_coefs0.tolist()+[5000,1]
    # print(wavcal_nloglike_poly(paras0, x, spectrum, spec_err, phoenix_HR8799_func, atran_2d_func, deg_wvs, deg_cont))

    # print(paras0[-3])
    # paras0[-3] *= np.nanmean(spectrum)/np.nanmean(wavcal_model_poly(paras0[0:np.size(paras0)-1], x, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont))
    # print(paras0[-3])
    simplex_init_steps =  np.ones(np.size(paras0))
    # print(simplex_init_steps)
    # print(paras0)
    simplex_init_steps[0:deg_wvs+1] = dwv/4
    if fitstarrv is not None:
        simplex_init_steps[deg_wvs+1] = 1
        simplex_init_steps[deg_wvs+2] = 200
    else:
        simplex_init_steps[deg_wvs+1] = 200
    # print(np.diag(simplex_init_steps))
    # print(paras0.shape)
    initial_simplex = np.concatenate([paras0[None,:],paras0[None,:] + np.diag(simplex_init_steps)],axis=0)
    # print(initial_simplex)
    # print(initial_simplex.shape)
    # exit()
    res = minimize(lambda paras: wavcal_nloglike_poly(paras, x,spectrum,spec_err, instr_trans,phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), paras0, method="nelder-mead",
                           options={"xatol": 1e-10, "maxiter": 1e5,"initial_simplex":initial_simplex,"disp":False})
    out = res.x
    # # print(res)
    # # print(paras0)
    # print(out)
    # # plt.plot(x,wavcal_model_poly(paras0, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont),label="model0")
    # plt.plot(x,spectrum,label="data")
    # plt.fill_between(x,spectrum-spec_err,spectrum+spec_err,label="data err",alpha=0.5)
    # plt.plot(x,wavcal_model_poly(out, x,spectrum,spec_err,instr_trans, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv),label="model")
    # plt.legend()
    # # plt.figure(2)
    # # out[-1] = 3000
    # # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    # # out[-1] = 4000
    # # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    # # out[-1] = 5000
    # # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    # # out[-1] = 8000
    # # plt.plot(x, wavcal_model_poly(out, x,spectrum,spec_err, phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv), label="res {0}".format(out[-1]))
    # # plt.legend()
    # plt.show()

    spl = InterpolatedUnivariateSpline(x_knots, out[0:deg_wvs+1], k=3, ext=0)
    return spl(x),out,x_knots

if __name__ == "__main__":
    try:
        import mkl

        mkl.set_num_threads(1)
    except:
        pass

    # x5, x4, x3, x2, x1, x0
    polycoefs_wavsol_jason = np.array([[-6.347647789728822e-23, -8.763050043241049e-18, -1.2137086057712525e-13, 9.833295632348853e-10, 2.205533800453195e-05, 2.438475434083729],
    [-1.5156054451627633e-19, 8.603654109299864e-16, -1.80716247518748e-12, 2.238166690200087e-09, 2.1123747999061772e-05, 2.3625134057175177],
    [1.2043489185290055e-19, -6.07134563401576e-16, 1.017386463625886e-12, -9.26853752638751e-11, 2.1241653129345714e-05, 2.291160322496969],
    [3.968104754663601e-19, -1.3124938289815753e-15, 1.112571593986215e-12, 6.134905626295576e-10, 2.017316485869667e-05, 2.2242225934551327],
    [8.041271039507505e-20, -1.188553085654241e-15, 3.294776997678977e-12, -2.9632213749811383e-09, 2.1429663086447367e-05, 2.1608253921854454],
    [9.523371077287098e-20, 1.9255684438936675e-16, -1.7740532731738176e-12, 3.2497136280253542e-09, 1.785770822240155e-05, 2.10169669134709],
    [-1.1921491225993055e-20, -2.4710800082806276e-17, -4.745847155094822e-13, 2.324820920356842e-09, 1.6978632482708987e-05, 2.0455151483483864],
    [-1.0581425110234265e-20, 2.1172786131371726e-16, -7.916021064868458e-13, 1.4427033600147576e-09, 1.7857955925450578e-05, 1.9918890220516599],
    [-7.67607806109257e-19, 3.893162857192925e-15, -6.697099419050439e-12, 5.105147905651209e-09, 1.6235110771773918e-05, 1.9414551546740444]])
    # print(polycoefs_wavsol_jason.shape)
    # exit()

    numthreads = 30
    mypool = mp.Pool(processes=numthreads)

    # mode ="science"
    mode = "star"
    fiber_num = 2
    fiber2extract_num = 2
    # for fiber2extract_num in [2]:#np.arange(1,4):
    if 1:

        mykpicdir = "/scr3/jruffio/data/kpic/"
        # kap_And_dir = os.path.join(mykpicdir,"kap_And_20191107")
        # kap_And_dir = os.path.join(mykpicdir,"bet_Peg_20191108")
        # kap_And_dir = os.path.join(mykpicdir,"kap_And_20191013")
        # kap_And_dir = os.path.join(mykpicdir,"2M0746A_20191012")
        kap_And_dir = os.path.join(mykpicdir,"DH_Tau_20191108")

        wvs_per_orders = np.zeros((9,2048))
        for order_id in range(9):
            wvs = np.polyval(polycoefs_wavsol_jason[8 - order_id, :], np.arange(0, 2048))
            wvs_per_orders[order_id,:] = wvs

        out = os.path.join(os.path.join(mykpicdir, "kap_And_20191107"), "calib", "instr_trans_f{0}.fits".format(fiber2extract_num))
        hdulist = pyfits.open(out)
        instr_trans = hdulist[0].data


        hdulist = pyfits.open(os.path.join(kap_And_dir, "calib", "trace_calib_polyfit_fiber{0}.fits".format(fiber2extract_num)))
        science_lw = np.abs(hdulist[0].data[:, :, 1])
        # hdulist = pyfits.open(os.path.join(kap_And_dir, "calib", "trace_calib_smooth_fiber{0}.fits".format(fiber2extract_num)))
        # science_lw2 = hdulist[0].data[:, ::-1, 1]
        # print(hdulist[0].data.shape)
        # for k in range(9):
        #     plt.subplot(9,1,9-k)
        #     tmp_wvs = wvs_per_orders[order_id,:]
        #     wmp_dwvs = wvs_per_orders[order_id,0:2047]-wvs_per_orders[order_id,1:2048]
        #     wmp_dwvs = np.append(wmp_dwvs,wmp_dwvs[-1])
        #     plt.plot(tmp_wvs/(science_lw[k,:]*2.355*wmp_dwvs))
        #     # plt.plot(science_lw2[k,:])
        # # plt.tight_layout()
        # plt.show()
        wmp_dwvs = wvs_per_orders[:,1:2048]-wvs_per_orders[:,0:2047]
        wmp_dwvs = np.concatenate([wmp_dwvs,wmp_dwvs[:,-1][:,None]],axis=1)
        science_lw_wvunit = science_lw*wmp_dwvs
        line_width_func = interp1d(np.ravel(wvs_per_orders),np.ravel(science_lw_wvunit), bounds_error=False, fill_value=np.nan)
        pixel_width_func = interp1d(np.ravel(wvs_per_orders),np.ravel(wmp_dwvs), bounds_error=False, fill_value=np.nan)

        if 0:
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
                atran_line_widths = np.array(pd.DataFrame(line_width_func(atran_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                atran_pixel_widths = np.array(pd.DataFrame(pixel_width_func(atran_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
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
            out = os.path.join(kap_And_dir, "..","models","atran", "atran_spec_list.fits")
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=np.array(water_list)))
            out = os.path.join(kap_And_dir, "..","models","atran", "water_list.fits")
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=np.array(atran_wvs)))
            out = os.path.join(kap_And_dir, "..","models","atran", "atran_wvs.fits")
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()
            plt.show()
        else:
            out = os.path.join(kap_And_dir, "..","models","atran", "atran_spec_list.fits")
            hdulist = pyfits.open(out)
            atran_spec_list = hdulist[0].data
            out = os.path.join(kap_And_dir, "..","models","atran", "water_list.fits")
            hdulist = pyfits.open(out)
            water_list = hdulist[0].data
            out = os.path.join(kap_And_dir, "..","models","atran", "atran_wvs.fits")
            hdulist = pyfits.open(out)
            atran_wvs = hdulist[0].data
            atran_2d_func = interp2d(atran_wvs,water_list, np.array(atran_spec_list), bounds_error=False, fill_value=np.nan)
            # print(atran_2d_func(wvs_per_orders[3,:],1.5*np.ones(wvs_per_orders[3,:].shape)).shape)
            # for order_id in range(9):
            #     plt.plot(wvs_per_orders[order_id,:],atran_2d_func(wvs_per_orders[order_id,:],3000),label="interp")
            #     plt.plot(wvs_per_orders[order_id,:],atran_2d_func(wvs_per_orders[order_id,:],5000),label="interp")
            #     plt.plot(wvs_per_orders[order_id,:],atran_2d_func(wvs_per_orders[order_id,:],8000),label="interp")
            # plt.show()

        phoenix_folder = "/scr3/jruffio/data/kpic/models/phoenix/"
        phoenix_wv_filename = os.path.join(phoenix_folder, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
        with pyfits.open(phoenix_wv_filename) as hdulist:
            phoenix_wvs = hdulist[0].data / 1.e4
        crop_phoenix = np.where((phoenix_wvs > 1.8 - (2.6 - 1.8) / 2) * (phoenix_wvs < 2.6 + (2.6 - 1.8) / 2))
        phoenix_wvs = phoenix_wvs[crop_phoenix]
        if "bet_Peg" in kap_And_dir or "DH_Tau" in kap_And_dir:
            phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "bet_Peg" + "*.fits"))[0]
        else:# "kap_And" in kap_And_dir:
            phoenix_model_host_filename = glob(os.path.join(phoenix_folder, "kap_And" + "*.fits"))[0]
        with pyfits.open(phoenix_model_host_filename) as hdulist:
            phoenix_HR8799 = hdulist[0].data[crop_phoenix]
        print("convolving: " + phoenix_model_host_filename)

        phoenix_line_widths = np.array(pd.DataFrame(line_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
        phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
        phoenix_HR8799_conv = convolve_spectrum_line_width(phoenix_wvs,phoenix_HR8799,phoenix_line_widths,mypool=mypool)
        phoenix_HR8799_conv = convolve_spectrum_pixel_width(phoenix_wvs, phoenix_HR8799_conv, phoenix_pixel_widths,mypool=mypool)

        phoenix_HR8799_func = interp1d(phoenix_wvs, phoenix_HR8799_conv/np.nanmax(phoenix_HR8799_conv), bounds_error=False, fill_value=np.nan)
        # exit()

        # kap_And_spec_func = interp1d(wmod, planet_convspec, bounds_error=False, fill_value=np.nan)
        # interp1d
        # convolve_spectrum_line_width(atran_arr[2,:])
        # plt.plot(atran_arr[1,:],atran_arr[2,:])
        # plt.show()
        # print(atran_arr.shape)
        # exit()

        star_filelist = glob(os.path.join(kap_And_dir, "calib", "*_"+mode+"flux_extract_f{0}.fits".format(fiber_num)))
        # star_filelist = glob(os.path.join(kap_And_dir, "calib", "*_"+mode+"flux_extract_f*.fits"))
        # star_filelist = glob(os.path.join(kap_And_dir, "calib", "*0023*_"+mode+"flux_extract_f{0}.fits".format(fiber_num)))
        star_filelist.sort()
        print(len(star_filelist))
        print(star_filelist)
        # exit()

        spectra = np.zeros((len(star_filelist),9,2048))
        star_wvs = np.zeros((len(star_filelist),9,2048))
        star_rn = np.zeros((len(star_filelist),9,2048))
        for fileid,filename in enumerate(star_filelist):
            out = os.path.join(kap_And_dir, "calib", os.path.basename(filename))
            out = glob(out)[0]
            hdulist = pyfits.open(out)
            extract_paras = hdulist[0].data[:,:,:]
            out = os.path.join(kap_And_dir, "calib", os.path.basename(filename).replace("_"+mode+"flux_extract","_"+mode+"flux_extract_res"))
            out = glob(out)[0]
            hdulist = pyfits.open(out)
            residuals = hdulist[0].data

            for order_id in range(9):
                # vec = extract_paras[order_id,:,fiber2extract_num-1]
                # vec_lpf = np.array(pd.DataFrame(vec).rolling(window=301, center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                # vec_hpf = vec - vec_lpf
                # vec_hpf_std = mad_std(vec_hpf[np.where(np.isfinite(vec_hpf))])
                # where_nans = np.where(np.abs(vec_hpf) > 5 * vec_hpf_std)

                # vec[where_nans] = np.nan
                # vec_lpf[np.where(np.isnan(vec))] = np.nan
                # wherefinitevec = np.where(np.isfinite(vec))
                # polyfit_trace_calib[order_id, :, para_id] = np.polyval(
                #     np.polyfit(x[wherefinitevec], vec[wherefinitevec], 2), x)

                # print(extract_paras.shape)
                # print(spectra.shape)
                a = extract_paras[order_id,:,fiber2extract_num-1]
                # a[where_nans] = np.nan
                spectra[fileid,order_id,:] = a
                a = extract_paras[order_id,:,fiber2extract_num-1+3]
                # a[where_nans] = np.nan
                star_rn[fileid,order_id,:] = np.abs(a)

        spectra[:,:,0:50] = np.nan
        spectra[:,:,(2048-50)::] = np.nan
        star_rn[:,:,0:50] = np.nan
        star_rn[:,:,(2048-50)::] = np.nan

        # out = glob(os.path.join(kap_And_dir, "calib", "*_star_spectra_f{0}_ef{1}.fits".format(fiber2extract_num,fiber2extract_num)))[0]
        # hdulist = pyfits.open(out)
        # star_spectra = hdulist[0].data
        # star_med_spec = np.nanmedian(star_spectra, axis=0)

        # star_rn = star_rn/np.sqrt(star_med_spec[None,:,:])

        # spectra = spectra/np.nanmean(spectra,axis=2)[:,:,None]
        cp_spectra = copy(spectra)
        if (fiber2extract_num == 3 and mode == "science") or (mode=="star" and fiber_num == fiber2extract_num):
            med_spec = np.nanmedian(spectra,axis=0)
            for k in range(len(star_filelist)):
                for l in range(9):
                    wherefinite = np.where(np.isfinite(spectra[k, l, :])*np.isfinite(med_spec[l, :]))[0]
                    spectra[k, l, :] = spectra[k, l, :]*np.nansum(spectra[k, l, wherefinite]*med_spec[l, wherefinite])/np.nansum(spectra[k, l, wherefinite]**2)
                # spectra[k, :, :] = spectra[k, :, :] * np.nansum(spectra[k, :, :] * med_spec) / np.nansum(spectra[k, :, :]**2)
        med_spec = np.nanmedian(spectra,axis=0)


        cp_star_rn = copy(star_rn)
        # star_rn_med = np.tile(np.nanmedian(star_rn, axis=2)[:,:,None],(1,1,2048))
        # star_rn[np.where(star_rn > 2*star_rn_med)] = np.nan
        # cp_star_rn[np.where(star_rn > 2*star_rn_med)] = np.nan
        # spectra[np.where(star_rn > 2*star_rn_med)] = np.nan
        # cp_spectra[np.where(star_rn > 2*star_rn_med)] = np.nan

        if len(star_filelist) >= 2:
            star_rn = star_rn/np.nanmedian(star_rn,axis=2)[:,:,None]
            star_rn_med = np.nanmedian(star_rn,axis=0)
            star_rn_res = (star_rn - star_rn_med[None, :, :])
            star_rn_std = np.zeros(star_rn.shape)
            for k in range(len(star_filelist)):
                for l in range(9):
                    tmp = star_rn_res[k,l,:]
                    star_rn_std[k,l,:] = mad_std(tmp[np.where(np.isfinite(tmp))])
            # star_rn_std = np.tile(np.nanstd(star_rn_res, axis=2)[:, :, None], (1, 1, 2048))
            # star_rn_ratio = np.tile(np.nanmax(np.abs(star_rn_res) / star_rn_std, axis=0)[None, :, :],(len(star_filelist), 1, 1))
            star_rn_ratio = np.abs(star_rn_res) / star_rn_std
            star_rn[np.where(star_rn_ratio > 5)] = np.nan
            cp_star_rn[np.where(star_rn_ratio > 5)] = np.nan
            spectra[np.where(star_rn_ratio > 5)] = np.nan
            cp_spectra[np.where(star_rn_ratio > 5)] = np.nan
        med_spec = np.nanmedian(spectra, axis=0)

        # spectra[np.where(star_rn>2*star_rn_med)]=np.nan
        # star_rn[np.where(star_rn>2*star_rn_med)]=np.nan

        # star_rn_med = np.nanmedian(star_rn, axis=2)
        # star_rn_res = star_rn-star_rn_med[:,:,None]
        # star_rn_std = np.tile(np.nanstd(star_rn_res,axis=2)[:,:,None],(1,1,2048))
        # star_rn_ratio = np.tile(np.nanmax(np.abs(star_rn_res)/star_rn_std,axis=0)[None,:,:],(len(star_filelist),1,1))
        #
        # spectra[np.where(star_rn_ratio>3)]=np.nan

        # star_background_med = np.nanmedian(star_background, axis=2)
        # star_background_res = star_background-star_background_med[:,:,None]
        # star_background_std = np.tile(np.nanstd(star_background_res,axis=2)[:,:,None],(1,1,2048))
        # star_background_ratio = np.tile(np.nanmax(np.abs(star_background_res)/star_background_std,axis=0)[None,:,:],(len(star_filelist),1,1))
        #
        # spectra[np.where(star_background_ratio>3)]=np.nan

        # med_spec = np.nanmedian(spectra,axis=0)




        # if len(star_filelist)>=2:
        #     spectra_res = (spectra-med_spec[None,:,:])
        #     spectra_std= np.zeros(spectra.shape)
        #     for k in range(len(star_filelist)):
        #         for l in range(9):
        #             tmp = spectra_res[k,l,:]
        #             spectra_std[k,l,:] = mad_std(tmp[np.where(np.isfinite(tmp))])
        #     # spectra_std = np.tile(np.nanstd(spectra_res,axis=2)[:,:,None],(1,1,2048))
        #     spectra_ratio = np.tile(np.nanmax(np.abs(spectra_res)/spectra_std,axis=0)[None,:,:],(len(star_filelist),1,1))
        #     # spectra_ratio = np.abs(spectra_res)/spectra_std
        #     spectra[np.where(spectra_ratio>5)]=np.nan
        #     cp_spectra[np.where(spectra_ratio>5)]=np.nan
        # med_spec = np.nanmedian(spectra,axis=0)


        # plt.figure(1)
        # for fileid, filename in enumerate(star_filelist):
        #     order_id = 1
        #     if fileid >= 5:
        #         linestyle = "--"
        #     else:
        #         linestyle = "-"
        #     plt.plot( spectra[fileid, order_id, :], alpha=1, linewidth=1,linestyle=linestyle,
        #              label=os.path.basename(filename))
        #     # for order_id in range(9):
        #     #     plt.subplot(9, 1, 9-order_id )
        #     #     plt.plot(wvs_per_orders[order_id,:],spectra[fileid,order_id,:],alpha=1,linewidth=1,label=os.path.basename(filename))
        # plt.legend()
        # plt.show()

        if 0:
            print(len(star_filelist))
            # ['/scr3/jruffio/data/kpic/bet_Peg_20191108/calib/nspec191108_0020_starflux_extract_f2.fits',
            #  '/scr3/jruffio/data/kpic/bet_Peg_20191108/calib/nspec191108_0021_starflux_extract_f2.fits',
            #  '/scr3/jruffio/data/kpic/bet_Peg_20191108/calib/nspec191108_0022_starflux_extract_f2.fits',
            #  '/scr3/jruffio/data/kpic/bet_Peg_20191108/calib/nspec191108_0023_starflux_extract_f2.fits',
            #  '/scr3/jruffio/data/kpic/bet_Peg_20191108/calib/nspec191108_0024_starflux_extract_f2.fits', No
            #  '/scr3/jruffio/data/kpic/bet_Peg_20191108/calib/nspec191108_0025_starflux_extract_f2.fits',
            #  '/scr3/jruffio/data/kpic/bet_Peg_20191108/calib/nspec191108_0026_starflux_extract_f2.fits']

            # for k in range(len(star_filelist)):
            #     print(star_filelist[k])
            #     plt.plot(spectra[k,l,:])
            # plt.legend()
            # plt.show()
            #     # exit()
            #     # spectrum = cp_spectra[k,l,:]
            #     # spec_err = cp_star_rn[k,l,:]
            if 1:

                l = 4
                spectrum = np.nanmean(spectra, axis=0)[l,:]
                spectrum[np.where(spectrum==0)] = np.nan
                spec_err = np.nanmean(star_rn, axis=0)[l,:]
                x = np.arange(0,2048)
                spec_err = np.clip(spec_err,0.5*np.nanmedian(spec_err),np.inf)
                # plt.plot(x,spectrum)
                # plt.plot(x,spec_err)
                # plt.fill_between(x,spectrum-spec_err,spectrum+spec_err)
                # plt.show()
                # spec_err /= np.nanmax(spectrum)
                wvs0 = wvs_per_orders[l,:]

                deg_wvs = 5
                deg_cont = 5
                # if "bet_Peg" in kap_And_dir:
                #     fitstarrv = 7.99 + 18.746 # None
                # else:
                #     fitstarrv = None
                fitstarrv = None
                new_wvs,out,x_knots = _fit_wavcal_poly((x,wvs0,spectrum,spec_err,instr_trans[l,:], phoenix_HR8799_func,atran_2d_func,deg_wvs,deg_cont,fitstarrv))
                # np.dot(M2,out[0:deg_wvs+1]),out[0:deg_wvs+1],M2,out[deg_wvs+1]
                print(out)
                # water = out[-1]
                # star_rv = out[-2]
                # plt.subplot(1, 2, 1)
                # plt.plot(spectrum/np.nanmax(spectrum),label="spec")
                # star_rv = 7.99+18.746
                # c_kms = 299792.458
                # tmp = atran_2d_func(wvs0,water)*phoenix_HR8799_func(wvs0*(1-star_rv/c_kms))
                # plt.plot(tmp/np.nanmax(tmp),label="Jason's wavcal")
                # tmp = atran_2d_func(new_wvs,water)*phoenix_HR8799_func(new_wvs*(1-star_rv/c_kms))
                # plt.plot(tmp/np.nanmax(tmp),label="new wavcal")
                # plt.legend()
                # plt.subplot(1,2,2)
                # plt.plot((new_wvs-wvs0)/(0.3/2048))
                # plt.show()
                # print(out,residuals)
            exit()
        elif 0:
            spectrum = np.nanmean(spectra, axis=0)
            spectrum[np.where(spectrum==0)] = np.nan
            tmp_spec_err = np.nanmean(star_rn, axis=0)

            x = np.arange(0,2048)
            deg_wvs = 5
            deg_cont = 5
            # if "bet_Peg" in kap_And_dir:
            #     fitstarrv = 7.99 + 18.746 # None
            # else:
            #     fitstarrv = None
            fitstarrv = None

            wvs0_chunks = []
            spectrum_chunks = []
            spec_err_chunks = []
            instr_trans_chunks = []
            kl_list = []
            for l in range(9):
                kl_list.append(l)
                wvs0_chunks.append(wvs_per_orders[l,:])
                spectrum_chunks.append(spectrum[l,:])
                instr_trans_chunks.append(instr_trans[l,:])
                spec_err = tmp_spec_err[l,:]
                spec_err = np.clip(spec_err,0.5*np.nanmedian(spec_err),np.inf)
                spec_err_chunks.append(spec_err)
            outputs_list = mypool.map(_fit_wavcal_poly, zip(itertools.repeat(x),
                                                          wvs0_chunks,spectrum_chunks,spec_err_chunks,instr_trans_chunks,
                                                          itertools.repeat(phoenix_HR8799_func),
                                                          itertools.repeat(atran_2d_func),
                                                          itertools.repeat(deg_wvs),
                                                          itertools.repeat(deg_cont),
                                                          itertools.repeat(fitstarrv)))

            new_wvs_arr = np.zeros((9,2048))
            for kl, out in zip(kl_list, outputs_list):
                new_wvs,out,x_knots = out
                new_wvs_arr[kl,:] = new_wvs

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=new_wvs_arr))
            out = os.path.join(kap_And_dir, "calib", "wvs_combined_images_f{0}.fits".format(fiber2extract_num))
            print(out)
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()
            exit()
        else:
            x = np.arange(0,2048)
            deg_wvs = 5
            deg_cont = 5
            # if "bet_Peg" in kap_And_dir:
            #     fitstarrv = 7.99 + 18.746 # None
            # else:
            #     fitstarrv = None
            fitstarrv = None

            wvs0_chunks = []
            spectrum_chunks = []
            spec_err_chunks = []
            instr_trans_chunks = []
            kl_list = []
            for k in range(len(star_filelist)):
                for l in range(9):
                    kl_list.append([k,l])
                    wvs0_chunks.append(wvs_per_orders[l,:])
                    spectrum_chunks.append(cp_spectra[k,l,:])
                    instr_trans_chunks.append(instr_trans[l,:])
                    spec_err = cp_star_rn[k,l,:]
                    spec_err = np.clip(spec_err,0.5*np.nanmedian(spec_err),np.inf)
                    spec_err_chunks.append(spec_err)
            outputs_list = mypool.map(_fit_wavcal_poly, zip(itertools.repeat(x),
                                                          wvs0_chunks,spectrum_chunks,spec_err_chunks,instr_trans_chunks,
                                                          itertools.repeat(phoenix_HR8799_func),
                                                          itertools.repeat(atran_2d_func),
                                                          itertools.repeat(deg_wvs),
                                                          itertools.repeat(deg_cont),
                                                          itertools.repeat(fitstarrv)))

            new_wvs_arr = np.zeros((len(star_filelist),9,2048))
            for kl, out in zip(kl_list, outputs_list):
                new_wvs,out,x_knots = out
                new_wvs_arr[kl[0],kl[1],:] = new_wvs
        # print(spectra.shape)
        # exit()


        print(new_wvs_arr.shape)
        med_new_wvs = np.nanmedian(new_wvs_arr,axis=0)
        maxdev_new_wvs = np.max(np.abs(new_wvs_arr-med_new_wvs[None,:,:]),axis=0)

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=med_new_wvs))
        out = os.path.join(kap_And_dir, "calib", "wvs_f{0}.fits".format(fiber2extract_num))
        print(out)
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=new_wvs_arr))
        out = os.path.join(kap_And_dir, "calib", "wvs_all_f{0}.fits".format(fiber2extract_num))
        print(out)
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        median_new_wvs_arr = np.median(new_wvs_arr,axis=0)
        for l in range(new_wvs_arr.shape[1]):
            plt.subplot(9, 1, 9 - l)
            for k in range(new_wvs_arr.shape[0]):
                plt.plot(new_wvs_arr[k,l,:]-median_new_wvs_arr[l,:])

        plt.show()
    exit()