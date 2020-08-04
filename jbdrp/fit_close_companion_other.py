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
from utils.badpix import *
from utils.misc import *
from utils.spectra import *
from scipy import interpolate
from PyAstronomy import pyasl
from scipy.optimize import nnls
from scipy.optimize import lsq_linear
import csv

def LPFvsHPF(myvec,cutoff):
    myvec_cp = copy(myvec)
    #handling nans:
    wherenans = np.where(np.isnan(myvec_cp))
    window = int(round(np.size(myvec_cp)/(cutoff/2.)/2.))#cutoff
    tmp = np.array(pd.DataFrame(np.concatenate([myvec_cp, myvec_cp[::-1]], axis=0)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))
    myvec_cp_lpf = np.array(pd.DataFrame(tmp).rolling(window=window, center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[0:np.size(myvec), 0]
    myvec_cp[wherenans] = myvec_cp_lpf[wherenans]


    fftmyvec = np.fft.fft(np.concatenate([myvec_cp, myvec_cp[::-1]], axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec


    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan

    # plt.figure(10)
    # plt.plot(myvec_cp,label="fixed")
    # plt.plot(myvec,label="ori")
    # plt.plot(myvec_cp_lpf,label="lpf")
    # plt.plot(LPF_myvec,label="lpf fft")
    # plt.show()
    return LPF_myvec,HPF_myvec


def _fitRV(paras):
    vsini,wvs,science_spec_hpf,science_spec_lpf, sciencehost_spec_hpf, sciencehost_spec_lpf, science_err,\
    slit1_spec_hpf, dark1_spec_hpf, dark2_spec_hpf,\
    wvs4broadening,planet_convspec_broadsampling,A0_spec,phoenix_A0_func,phoenix_host_func,\
    A0_rv,A0_baryrv,host_rv,science_baryrv,c_kms,cutoff,rv_list = paras
    print("working on ",vsini)

    if vsini != 0:
        planet_broadspec = pyasl.rotBroad(wvs4broadening, planet_convspec_broadsampling, 0.5, vsini)
        planet_broadspec_func = interp1d(wvs4broadening, planet_broadspec, bounds_error=False,
                                         fill_value=np.nan)
    else:
        planet_broadspec_func = interp1d(wvs4broadening, planet_convspec_broadsampling, bounds_error=False,
                                         fill_value=np.nan)


    where_data_nans = np.where(np.isnan(science_spec_hpf))
    transmission = A0_spec / phoenix_A0_func(wvs * (1 - (A0_rv - A0_baryrv) / c_kms))
    transmission[where_data_nans] = np.nan
    m1_norv = planet_broadspec_func(wvs* (1 - (-15 - science_baryrv) / c_kms)) * transmission
    # print(np.nanmean(m1_norv), np.nanmean(science_spec_lpf))
    # exit()
    m1_norv = m1_norv / np.nanmean(m1_norv) * np.nanmean(science_spec_lpf)
    m1_norv_spec_hpf = np.zeros(science_spec_lpf.shape)
    for order_id in range(Norders):
        _, m1_norv_spec_hpf[order_id, :] = LPFvsHPF(m1_norv[order_id, :], cutoff=cutoff)

    fluxout = np.zeros((2,6,np.size(rv_list)))
    dAICout = np.zeros((6,np.size(rv_list)))
    logpostout = np.zeros((6,np.size(rv_list)))

    data_hpf_list = [science_spec_hpf, m1_norv_spec_hpf, slit1_spec_hpf, dark1_spec_hpf, dark2_spec_hpf]
    for data_id, data_hpf in enumerate(data_hpf_list):

        where_data_nans = np.where(np.isnan(data_hpf))
        transmission = A0_spec / phoenix_A0_func(wvs * (1 - (A0_rv - A0_baryrv) / c_kms))
        transmission[where_data_nans] = np.nan

        m2 = np.zeros(science_spec_lpf.shape)
        for order_id in range(Norders):
            m2[order_id, :] = LPFvsHPF(sciencehost_spec_hpf[order_id, :] / sciencehost_spec_lpf[order_id, :] * science_spec_lpf[order_id, :], cutoff=cutoff)[1]
        m3 = phoenix_host_func(wvs * (1 - (host_rv - science_baryrv) / c_kms)) * transmission
        for order_id in range(Norders):
            m3_tmp_lpf, m3_tmp_hpf = LPFvsHPF(m3[order_id, :], cutoff=cutoff)
            m3[order_id, :] = m3_tmp_hpf / m3_tmp_lpf * science_spec_lpf[order_id, :]

        m2_arr = np.zeros((m2.shape[0], m2.shape[0], m2.shape[1]))
        m3_arr = np.zeros((m2.shape[0], m2.shape[0], m2.shape[1]))
        for order_id in range(m2.shape[0]):
            m2_arr[order_id, order_id, :] = m2[order_id, :]
            m3_arr[order_id, order_id, :] = m3[order_id, :]
        # plt.plot(np.ravel(m2),label="m2")
        # plt.plot(np.ravel(m3),label="m3")
        # plt.legend()
        # plt.show()
        b1 = np.zeros((m2.shape[0],m2.shape[0],m2.shape[1]))
        b2 = np.zeros((m2.shape[0],m2.shape[0],m2.shape[1]))
        b3 = np.zeros((m2.shape[0],m2.shape[0],m2.shape[1]))
        for order_id in range(m2.shape[0]):
            b1[order_id,order_id,:] = 1
            b2[order_id,order_id,:] = np.arange(m2.shape[1])
            b3[order_id,order_id,:] = np.arange(m2.shape[1])**2

        for rv_id, rv in enumerate(rv_list):
            # print(vsini_id,np.size(vsini_list),rv_id,np.size(rv_list))
            m1 = planet_broadspec_func(wvs * (1 - (rv - science_baryrv) / c_kms)) * transmission
            # m1  = m1/np.nanmean(m1)*np.nanmean(A0_spec)
            for order_id in range(Norders):
                m1[order_id, :] = LPFvsHPF(m1[order_id, :], cutoff=cutoff)[1]
            # science_spec_hpf[np.where(np.isnan(m1)*np.isnan(m2))] = np.nan

            ravelHPFdata = np.ravel(data_hpf)
            ravelwvs = np.ravel(wvs)

            # plt.figure(1)
            # plt.plot(np.ravel(wvs), np.ravel(transmission))
            # plt.figure(2)
            # plt.plot(np.ravel(wvs), np.ravel(m1))
            # plt.figure(3)
            # plt.plot(np.ravel(wvs), ravelHPFdata)
            # plt.show()

            where_data_finite = np.where(np.isfinite(ravelHPFdata*np.ravel(science_err)))
            ravelHPFdata = ravelHPFdata[where_data_finite]
            ravelwvs = ravelwvs[where_data_finite]
            sigmas_vec = np.ravel(science_err)[where_data_finite]
            logdet_Sigma = np.sum(2 * np.log(sigmas_vec))

            if 0:
                m1_ravel = np.ravel(m1)[where_data_finite]
                m2_ravel = np.ravel(m2)[where_data_finite]
                m3_ravel = np.ravel(m3)[where_data_finite]

                b1_ravel = np.reshape(b1,(m2.shape[0],m2.shape[0]*m2.shape[1]))[:,where_data_finite[0]].T
                b2_ravel = np.reshape(b2,(m2.shape[0],m2.shape[0]*m2.shape[1]))[:,where_data_finite[0]].T
                b3_ravel = np.reshape(b3,(m2.shape[0],m2.shape[0]*m2.shape[1]))[:,where_data_finite[0]].T

                m3_ravel = m3_ravel - m2_ravel*np.sum(m2_ravel*m3_ravel)/np.sum(m2_ravel**2)
                HPFmodel = np.concatenate([m1_ravel[:, None], m2_ravel[:, None], m3_ravel[:, None], b1_ravel], axis=1)
                HPFmodel_H0 = np.concatenate([m2_ravel[:, None], m3_ravel[:, None], b1_ravel], axis=1)
                # HPFmodel = np.concatenate([m1_ravel[:, None], m2_ravel[:, None], m3_ravel[:, None], b1_ravel,b2_ravel,b3_ravel], axis=1)
                # HPFmodel_H0 = np.concatenate([m2_ravel[:, None], m3_ravel[:, None], b1_ravel,b2_ravel,b3_ravel], axis=1)
                # HPFmodel = m1_ravel[:,None]
            else:

                m1_ravel = np.ravel(m1)[where_data_finite]
                m2_ravel = np.reshape(m2_arr,(m2.shape[0],m2.shape[0]*m2.shape[1]))[:,where_data_finite[0]].T#np.ravel(m2)[where_data_finite]
                m3_ravel = np.reshape(m3_arr,(m2.shape[0],m2.shape[0]*m2.shape[1]))[:,where_data_finite[0]].T#np.ravel(m3)[where_data_finite]

                b1_ravel = np.reshape(b1,(m2.shape[0],m2.shape[0]*m2.shape[1]))[:,where_data_finite[0]].T
                # b2_ravel = np.reshape(b2,(m2.shape[0],m2.shape[0]*m2.shape[1]))[:,where_data_finite[0]].T
                # b3_ravel = np.reshape(b3,(m2.shape[0],m2.shape[0]*m2.shape[1]))[:,where_data_finite[0]].T

                for order_id in range(Norders):
                    m3_ravel[order_id, :] = m3_ravel[order_id, :] - m2_ravel[order_id, :]*np.sum(m2_ravel[order_id, :]*m3_ravel[order_id, :])/np.sum(m2_ravel[order_id, :]**2)
                HPFmodel = np.concatenate([m1_ravel[:, None], m2_ravel,m3_ravel, b1_ravel], axis=1)
                HPFmodel_H0 = np.concatenate([m2_ravel,m3_ravel, b1_ravel], axis=1)
                # HPFmodel = np.concatenate([m1_ravel[:, None]], axis=1)
                # HPFmodel_H0 = np.concatenate([m2_ravel], axis=1)


            # print(np.where(np.isnan(sigmas_vec)))
            # print(np.sum(ravelHPFdata * m1_ravel / sigmas_vec ** 2) / np.sum((m1_ravel / sigmas_vec) ** 2))
            # print(np.sum(ravelHPFdata * m1_ravel / sigmas_vec ** 2) , np.sum((m1_ravel / sigmas_vec) ** 2))
            # print(np.sum(ravelHPFdata) , np.sum((m1_ravel) ** 2))
            # exit()

            # # # print(np.where(np.isnan( ravelHPFdata/sigmas_vec)))
            # # # print(np.where(np.isnan( HPFmodel/sigmas_vec[:,None])))
            # # plt.plot(HPFmodel[:,0]/np.std(HPFmodel[:,0]),color="blue")
            # # plt.plot(HPFmodel[:,1]/np.std(HPFmodel[:,1]),color="red")
            # plt.plot(ravelHPFdata/np.std(ravelHPFdata),color="purple")
            # plt.plot(HPFmodel_H0/np.std(HPFmodel_H0),"--",color="green")
            # a= np.ravel(sciencehost_spec_hpf)[where_data_finite]
            # plt.plot(a/np.std(a),"--",color="red")
            # plt.show()

            if 1:
                # HPFparas, HPFchi2, rank, s = np.linalg.lstsq(HPFmodel / sigmas_vec[:, None], ravelHPFdata / sigmas_vec,
                #                                              rcond=None)
                # HPFparas_H2, HPFchi2_H2, rank, s = np.linalg.lstsq(HPFmodel_H2 / sigmas_vec[:, None],
                #                                                    ravelHPFdata / sigmas_vec, rcond=None)
                # HPFparas_H0, HPFchi2_H0, rank, s = np.linalg.lstsq(HPFmodel_H0 / sigmas_vec[:, None],
                #                                                    ravelHPFdata / sigmas_vec, rcond=None)
                # print(HPFparas, HPFparas_H0)  # [0.00113497 0.59605852] [0.00378131] [0.67891099]

                # HPFparas,HPFchi2_H0 = nnls(HPFmodel / sigmas_vec[:, None],ravelHPFdata / sigmas_vec)
                # HPFparas_H0,HPFchi2_H0 = nnls(HPFmodel_H0 / sigmas_vec[:, None],ravelHPFdata / sigmas_vec)

                # bounds_min = [-np.inf,]*HPFmodel.shape[1]
                # bounds_max = [np.inf,]*HPFmodel.shape[1]
                # bounds_min[1] = 0
                # # bounds_min[2] = 0
                norm_HPFmodel = HPFmodel/sigmas_vec[:, None]
                norm_HPFmodel_H0 = HPFmodel_H0/sigmas_vec[:, None]
                # HPFparas = lsq_linear(norm_HPFmodel,ravelHPFdata / sigmas_vec,bounds=(bounds_min,bounds_max)).x
                # HPFparas_H0 = lsq_linear(norm_HPFmodel_H0,ravelHPFdata / sigmas_vec,bounds=(bounds_min[1::],bounds_max[1::])).x
                HPFparas = lsq_linear(norm_HPFmodel,ravelHPFdata / sigmas_vec).x
                HPFparas_H0 = lsq_linear(norm_HPFmodel_H0,ravelHPFdata / sigmas_vec).x

                data_model = np.dot(HPFmodel, HPFparas)
                data_model_H0 = np.dot(HPFmodel_H0, HPFparas_H0)
                deltachi2 = 0  # chi2ref-np.sum(ravelHPFdata**2)
                ravelresiduals = ravelHPFdata - data_model
                ravelresiduals_H0 = ravelHPFdata - data_model_H0
                HPFchi2 = np.nansum((ravelresiduals/sigmas_vec) ** 2)
                HPFchi2_H0 = np.nansum((ravelresiduals_H0/sigmas_vec) ** 2)


                Npixs_HPFdata = HPFmodel.shape[0]
                minus2logL_HPF = Npixs_HPFdata * (
                            1 + np.log(HPFchi2 / Npixs_HPFdata) + logdet_Sigma + np.log(2 * np.pi))
                minus2logL_HPF_H0 = Npixs_HPFdata * (
                            1 + np.log(HPFchi2_H0 / Npixs_HPFdata) + logdet_Sigma + np.log(2 * np.pi))
                AIC_HPF = 2 * (HPFmodel.shape[-1]) + minus2logL_HPF
                AIC_HPF_H0 = 2 * (HPFmodel_H0.shape[-1]) + minus2logL_HPF_H0

                # covphi = HPFchi2 / Npixs_HPFdata * np.linalg.inv(np.dot(norm_HPFmodel.T, norm_HPFmodel))
                slogdet_icovphi0 = np.linalg.slogdet(np.dot(norm_HPFmodel.T, norm_HPFmodel))

                a = np.zeros(np.size(HPFparas))
                a[0] = 1
                a_err = np.sqrt(lsq_linear(np.dot(norm_HPFmodel.T, norm_HPFmodel)/(HPFchi2 / Npixs_HPFdata),a).x[0])

                fluxout[0,data_id, rv_id] = HPFparas[0]
                fluxout[1,data_id, rv_id] = a_err
                dAICout[data_id, rv_id] = AIC_HPF_H0-AIC_HPF
                logpostout[data_id, rv_id] = -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0[1] - (
                            Npixs_HPFdata - 1 + 2 - 1) / (2) * np.log(HPFchi2)

                # # exit()
                # # print("H1",HPFparas)
                # print(slogdet_icovphi0)
                # print(logpostout[data_id, rv_id] )
                # print("H0",HPFparas_H0)
                # print("H1",HPFparas)
                # # print(covphi)
                # print(HPFparas[0]/a_err,HPFparas[0],a_err,(HPFchi2 / Npixs_HPFdata))
                # # plt.fill_between(ravelwvs,ravelHPFdata-sigmas_vec,ravelHPFdata+sigmas_vec, label = "error")
                # plt.subplot(2,1,1)
                # plt.plot(ravelwvs,ravelHPFdata,label = "data",alpha= 1,color="red")
                # # plt.plot(ravelwvs,sigmas_vec,label = "sig",alpha= 0.5)
                # plt.plot(ravelwvs,HPFparas[0]*m1_ravel, label = "planet",alpha= 0.5,color="blue")
                # # plt.plot(ravelwvs,HPFparas[1]*m2_ravel+HPFparas[2]*m3_ravel, label = "star",alpha= 0.5)
                # plt.plot(ravelwvs,data_model, label = "model",alpha= 1,color="green")
                # # plt.plot(ravelwvs,HPFparas[2]*m3_ravel, label = "star",alpha= 0.5)
                # # plt.plot(ravelwvs,HPFparas_H0[0]*m2_ravel, label = "star m2",alpha= 0.5)
                # # plt.plot(ravelwvs,HPFparas_H0[1]*m3_ravel, label = "star m3",alpha= 0.5)
                # # plt.plot(ravelwvs,ravelHPFdata-HPFparas[1]*m2_ravel, label = "res",alpha= 0.5)
                # # plt.plot(ravelwvs,ravelHPFdata-HPFparas_H0[0]*m2_ravel-HPFparas_H0[1]*m3_ravel, label = "res h0",alpha= 0.5)
                # plt.legend()
                # plt.subplot(2,1,2)
                # plt.fill_between(ravelwvs,-sigmas_vec,sigmas_vec,alpha=0.5,color="gray")
                # plt.plot(ravelwvs,ravelHPFdata,label = "data",alpha= 0.5,color="red")
                # plt.plot(ravelwvs,ravelresiduals,label = "res",alpha= 0.5,color="green")
                # # print(np.nanstd(ravelresiduals))
                # # print(np.nanstd(ravelresiduals[0:np.size(ravelresiduals)//2]),np.nanstd(ravelresiduals[np.size(ravelresiduals)//2::]))
                # # print(np.median(sigmas_vec[0:np.size(ravelresiduals)//2]),np.median(sigmas_vec[np.size(ravelresiduals)//2::]))
                # plt.legend()
                # plt.show()
                # # exit()
    return vsini,fluxout,dAICout,logpostout

if __name__ == "__main__":
    try:
        import mkl

        mkl.set_num_threads(1)
    except:
        pass


    mykpicdir = "/scr3/jruffio/data/kpic/"
    phoenix_folder = "/scr3/jruffio/data/kpic/models/phoenix/"

    fib = 1 # HERE!!!!
    hostfib = 2 # HERE!!!!
    numthreads = 11
    # Combining oders 2M0746A 9 / np.sqrt(np.sum(1 / np.array([7, 8, 6, 7, 4, 4, 13.5, 7, 6]) ** 2))
    # selec_orders = [0,6] # HERE!!!!
    # selec_orders = [0,6]
    selec_orders = [0,1,2,5,6,7]
    # selec_orders = [0,1,2]
    # selec_orders = [0,1,2,3,4,5,6,7,8]
    # selec_orders = [1,2,6,7,8]
    Norders=len(selec_orders)
    cutoff = 40
    c_kms = 299792.458
    vsini_list = np.linspace(0,100,11,endpoint=True) # HERE!!!!
    # vsini_list = np.array([40]) # HERE!!!!
    # rv_list = np.concatenate([np.arange(-400, -10, 5), np.arange(-10, 10, 0.1), np.arange(10, 400, 5)], axis=0)
    rv_list = np.linspace(-400,400,101,endpoint=True)
    # rv_list = np.linspace(-1.0,400,1)
    save = True
    plotonly = False
    # [[-1.80972972e+01 - 1.49143101e+01 - 2.17664527e+01 - 3.42045802e+01
    #   - 3.70393796e+01 - 4.14921048e+01 - 4.15031564e+01 - 4.89345879e+01
    #   - 5.79634955e+01]
    #  [-9.35020989e+00 - 2.48176362e+00 - 4.22284978e+00 - 8.28578552e+00
    #   - 1.11541973e+01 - 1.42992317e+01 - 2.05399413e+01 - 2.72518274e+01
    #   - 3.57545284e+01]
    #  [1.54382410e+03  2.01119527e+03  3.00495446e+03  3.84622960e+03
    #  3.56563466e+03  3.42191068e+03  3.06027832e+03  2.82196257e+03
    #  2.20042440e+03]]
    # ['/scr3/jruffio/data/
    if 1:
        ## science selection
        # sciencedir = os.path.join(mykpicdir, "20191107_kap_And_B") # HERE!!!!
        # sciencedir = os.path.join(mykpicdir, "20200608_kap_And_B")
        # phoenix_host_filename = glob(os.path.join(phoenix_folder, "kap_And" + "*.fits"))[0] # HERE!!!!
        # host_rv = -12.7 # HERE!!!!
        # sciencedir = os.path.join(mykpicdir, "20200608_HR_7672_B") # Teff = 6000 logg 4.5
        sciencedir = os.path.join(mykpicdir, "20200609_HR_7672_B") # Teff = 6000 logg 4.5
        phoenix_host_filename = glob(os.path.join(phoenix_folder, "HR_7672_lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"))[0]
        host_rv = -5 # HERE!!!!
        filelist = glob(os.path.join(sciencedir, "*fluxes.fits"))
        filelist.sort()
        print(len(filelist))
    if 1:
        combined = True
        science_spec,science_err,slit_spec,dark_spec,science_baryrv = combine_spectra_from_folder(filelist,"science")
        hdulist = pyfits.open(filelist[0])
        science_mjd = None#dulist[0].header["MJD"]
    # for sciencefilename in filelist:
    #     print(sciencefilename)
    #     combined = False
    #     science_spec,science_err,slit_spec,dark_spec,science_baryrv = combine_spectra_from_folder([sciencefilename],"science")
    #     hdulist = pyfits.open(sciencefilename)
    #     science_mjd = hdulist[0].header["MJD"]

        print(science_spec.shape)
        print(np.nanmean(science_spec,axis=2))
        # exit()

        print(science_err.shape)

        ## standard star selection
        # A0dir = os.path.join(mykpicdir,"20191107_kap_And") # HERE!!!!
        # A0dir = os.path.join(mykpicdir,"20200608_kap_And")
        # A0_rv = -12.7 #km/s
        # A0dir = os.path.join(mykpicdir,"20200608_zet_Aql")
        A0dir = os.path.join(mykpicdir,"20200609_zet_Aql")
        A0_rv = -25 #km/s
        phoenix_A0_filename = glob(os.path.join(phoenix_folder, "kap_And" + "*.fits"))[0]
        filelist = glob(os.path.join(A0dir, "*fluxes.fits"))
        filelist.sort()
        print(filelist)
        A0_spec,A0_err,_,_,A0_baryrv = combine_spectra_from_folder(filelist,"star",science_mjd=science_mjd)
        cp_A0_spec = copy(edges2nans(A0_spec))
        cp_A0_err = copy(edges2nans(A0_err))

        if not plotonly:
            A0_spec,A0_err = cp_A0_spec[fib,selec_orders],cp_A0_err[fib,selec_orders]
            science_spec = edges2nans(science_spec)
            science_err = edges2nans(science_err)
            sciencehost_spec, sciencehost_err = science_spec[hostfib,selec_orders],science_err[hostfib,selec_orders]
            science_spec, science_err = science_spec[fib,selec_orders],science_err[fib,selec_orders]
            slit1_spec, dark1_spec,dark2_spec = slit_spec[fib,selec_orders],dark_spec[fib,selec_orders],dark_spec[fib+4,selec_orders]
            where_nans = np.where(np.isnan(A0_spec)+np.isnan(science_spec))
            science_spec[where_nans] = np.nan
            science_err[where_nans] = np.nan
            dark1_spec[where_nans] = np.nan
            dark2_spec[where_nans] = np.nan
            slit1_spec[where_nans] = np.nan


            hdulist = pyfits.open(glob(os.path.join(sciencedir, "calib", "*_wvs.fits"))[0])
            wvs = hdulist[0].data[fib,selec_orders,:]
            wvs_host = hdulist[0].data[hostfib,selec_orders,:]

            for k in range(Norders):
                tmp_sciencehist_spec =  np.array(pd.DataFrame(sciencehost_spec[k,:]).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                f =  interp1d(wvs_host[k,:], tmp_sciencehist_spec, bounds_error=False,fill_value=np.nan)
                sciencehost_spec[k, :] = f(wvs[k,:])
            sciencehost_spec[where_nans] = np.nan

            # print(wvs.shape,science_spec.shape)
            # for k in range(Norders):
            #     plt.subplot(Norders,1,Norders-k)
            #     # plt.fill_between(wvs[k,:], science_spec[k,:]-science_err[k,:], science_spec[k,:]+science_err[k,:],label="Error bars",color="orange")
            #     plt.plot(wvs[k,:], science_spec[k,:],label="spec",color="blue",linewidth=0.5)
            #     plt.plot(wvs[k,:], sciencehost_spec[k,:],label="sciencehost",color="red",linewidth=0.5)
            #     # plt.plot(wvs[k,:], sciencehost_spec[k,:]-A0_spec[k,:]/np.nanmean(A0_spec[k,:])*np.nanmean(sciencehost_spec[k,:]),label="A0",color="green",linewidth=0.5,linestyle="--")
            #     plt.plot(wvs[k,:], A0_spec[k,:]/np.nanmean(A0_spec[k,:])*np.nanmean(sciencehost_spec[k,:]),label="A0",color="green",linewidth=0.5,linestyle="--")
            #     # plt.fill_between(wvs[k,:], host_spec[k,:]-host_err[k,:], host_spec[k,:]+host_err[k,:],label="Error bars",color="orange")
            #     # plt.plot(wvs[k,:], host_spec[k,:],label="spec",color="blue",linewidth=0.5)
            #     # plt.plot(wvs[k,:], slit1_spec[k,:],label="slit background 1",alpha=0.5,color="grey")
            #     # plt.plot(wvs[k,:], dark1_spec[k,:],label="dark background 1",alpha=0.5,color="grey")
            #     # plt.plot(wvs[k,:], slit2_spec[k,:],label="slit background 2",alpha=0.5,linestyle="--",color="grey")
            #     # plt.plot(wvs[k,:], dark2_spec[k,:],label="dark background 2",alpha=0.5,linestyle="--",color="grey")
            #     # plt.ylim([0-10*np.nanstd(slit1_spec[k,:]),np.nanmax(science_spec[k,:])+5*np.nanmedian(science_err[k,:])])
            # plt.legend()
            # plt.show()


            line_width_filename = glob(os.path.join(sciencedir, "calib", "*_line_width_smooth.fits"))[0]
            hdulist = pyfits.open(line_width_filename)
            line_width = hdulist[0].data[fib,selec_orders,:]
            dwvs = wvs[:, 1:2048] - wvs[:, 0:2047]
            dwvs = np.concatenate([dwvs, dwvs[:, -1][:, None]], axis=1)
            line_width_wvunit = line_width[:, :] * dwvs
            line_width_func = interp1d(np.ravel(wvs), np.ravel(line_width_wvunit), bounds_error=False,
                                       fill_value=np.nan)
            pixel_width_func = interp1d(np.ravel(wvs), np.ravel(dwvs), bounds_error=False, fill_value=np.nan)


            specpool = mp.Pool(processes=numthreads)
            if "kap_And_B" in sciencedir or "DH_Tau_B" in sciencedir or "HR_7672_B" in sciencedir: # HERE!!!!
                if 0:
                    # travis_spec_filename = os.path.join("/scr3/jruffio/data/kpic/models/planets_templates/","lte2048-3.77-0.11.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019_COscl=1.00_H2Oscl=1.00_CH4scl=1.0_4KPIC.7_D2E")
                    # file1 = open(travis_spec_filename, 'r')
                    # file2 = open(travis_spec_filename.replace(".7",".7_D2E"), 'w')
                    # for k,line in enumerate(file1):
                    #     file2.write(line.replace("D","E"))
                    #     # if "0 9" in line[88:91]:
                    #     #     file2.write(line.replace("0 9","009"))
                    #     # else:
                    #     #     file2.write(line)

                    # travis_spec_filename = os.path.join("/scr3/jruffio/data/kpic/models/planets_templates/","lte2048-3.77-0.11.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019_COscl=1.00_H2Oscl=1.00_CH4scl=1.0_4KPIC.7_D2E_Ksorted")
                    travis_spec_filename = os.path.join("/scr3/jruffio/data/kpic/models/planets_templates/",
                                                        "lte2048-3.77-0.11.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019_COscl=1.00_H2Oscl=1.00_CH4scl=1.0_4KPIC.7_D2E_Ksorted_conv")
                    if len(glob(travis_spec_filename)) == 0:
                        #/scr3/jruffio/data/kpic/models/planets_templates/lte2048-3.7_D2E7-0.11.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019_COscl=1.00_H2Oscl=1.00_CH4scl=1.0_4KPIC.7_D2E
                        #/scr3/jruffio/data/kpic/models/planets_templates/lte2048-3.77-0.11.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019_COscl=1.00_H2Oscl=1.00_CH4scl=1.0_4KPIC.7_D2E
                        data = np.loadtxt(os.path.join("/scr3/jruffio/data/kpic/models/planets_templates/","lte2048-3.77-0.11.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019_COscl=1.00_H2Oscl=1.00_CH4scl=1.0_4KPIC.7_D2E"))
                        print(data.shape)
                        wmod = data[:, 0] / 10000.
                        wmod_argsort = np.argsort(wmod)
                        wmod = wmod[wmod_argsort]
                        crop_moltemp = np.where(
                            (wmod > wvs[0,0] - (wvs[0,-1] - wvs[0,0]) / 2) * (wmod < wvs[-1,-1] + (wvs[-1,-1] - wvs[-1,0]) / 2))
                        wmod = wmod[crop_moltemp]
                        mol_temp = data[wmod_argsort, 1][crop_moltemp]
                        mol_temp = 10**(mol_temp-np.max(mol_temp))

                        if 1:
                            pl_line_widths = np.array(pd.DataFrame(line_width_func(wmod)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                            pl_pixel_widths = np.array(pd.DataFrame(pixel_width_func(wmod)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                            planet_convspec = convolve_spectrum_line_width(wmod, mol_temp, pl_line_widths,mypool=specpool)
                            planet_convspec = convolve_spectrum_pixel_width(wmod, planet_convspec, pl_pixel_widths,mypool=specpool)
                        # import matplotlib.pyplot as plt
                        # plt.plot(wmod,mol_temp)#,data[::100,1])
                        # print(mol_temp.shape)
                        # plt.show()
                        # exit()
                        # print("convolving: " + mol_template_filename)
                        # planet_convspec = convolve_spectrum(wmod, mol_temp, R, specpool)

                        with open(travis_spec_filename, 'w+') as csvfile:
                            csvwriter = csv.writer(csvfile, delimiter=' ')
                            csvwriter.writerows([["wvs", "spectrum"]])
                            csvwriter.writerows([[a, b] for a, b in zip(wmod, planet_convspec)])
                        plt.plot(wmod, planet_convspec)
                        plt.show()
                    #
                    with open(travis_spec_filename, 'r') as csvfile:
                        csv_reader = csv.reader(csvfile, delimiter=' ')
                        list_starspec = list(csv_reader)
                        oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                        col_names = oriplanet_spec_str_arr[0]
                        ori_planet_spec = oriplanet_spec_str_arr[1::3, 1].astype(np.float)
                        wmod = oriplanet_spec_str_arr[1::3, 0].astype(np.float)
                        ori_planet_spec = 10**(ori_planet_spec-np.max(ori_planet_spec))

                        crop_plmodel = np.where((wmod>1.8-(2.6-1.8)/2)*(wmod<2.6+(2.6-1.8)/2))
                        wmod = wmod[crop_plmodel]
                        ori_planet_spec = ori_planet_spec[crop_plmodel]
                        ori_planet_spec /= np.nanmean(ori_planet_spec)
                        science_model_spline = interpolate.splrep(wmod, ori_planet_spec)
                        # # plt.plot(wmod, ori_planet_spec)
                        # # print(wmod[np.size(wmod)//2]/(wmod[np.size(wmod)//2+1]-wmod[np.size(wmod)//2]))
                        # if 1:
                        #     pl_line_widths = np.array(pd.DataFrame(line_width_func(wmod)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                        #     pl_pixel_widths = np.array(pd.DataFrame(pixel_width_func(wmod)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                        #     planet_convspec = convolve_spectrum_line_width(wmod, ori_planet_spec, pl_line_widths,mypool=specpool)
                        #     planet_convspec = convolve_spectrum_pixel_width(wmod, planet_convspec, pl_pixel_widths,mypool=specpool)
                        # planet_convspec /= np.nanmean(planet_convspec)
                        # print("convolving: " + travis_spec_filename)
                        # # kap_And_spec_func = interp1d(wmod, planet_convspec, bounds_error=False, fill_value=np.nan)
                        # science_model_spline = interpolate.splrep(wmod, planet_convspec)

                    # plt.plot(wmod, planet_convspec)
                    # plt.show()
                    # exit()

                elif 1:
                    travis_spec_filename = os.path.join("/scr3/jruffio/data/kpic/models/planets_templates/",
                                                        "KapAnd_lte19-3.50-0.0.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019.7.save")
                    import scipy.io as scio
                    travis_spectrum = scio.readsav(travis_spec_filename)
                    ori_planet_spec = np.array(travis_spectrum["f"])
                    wmod = np.array(travis_spectrum["w"]) / 1.e4
                    crop_plmodel = np.where((wmod>1.8-(2.6-1.8)/2)*(wmod<2.6+(2.6-1.8)/2))
                    wmod = wmod[crop_plmodel]
                    ori_planet_spec = ori_planet_spec[crop_plmodel]
                    # print(wmod)
                    # plt.plot(wmod,ori_planet_spec)
                    # plt.show()
                    # planet_convspec = convolve_spectrum(wmod, ori_planet_spec, 30000, specpool)
                    if 1:
                        pl_line_widths = np.array(pd.DataFrame(line_width_func(wmod)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                        pl_pixel_widths = np.array(pd.DataFrame(pixel_width_func(wmod)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                        planet_convspec = convolve_spectrum_line_width(wmod,ori_planet_spec,pl_line_widths,mypool=specpool)
                        planet_convspec = convolve_spectrum_pixel_width(wmod,planet_convspec,pl_pixel_widths,mypool=specpool)
                    planet_convspec /= np.nanmean(planet_convspec)
                    print("convolving: " + travis_spec_filename)
                    # kap_And_spec_func = interp1d(wmod, planet_convspec, bounds_error=False, fill_value=np.nan)
                    science_model_spline = interpolate.splrep(wmod, planet_convspec)
                    # exit()

                else:
                    with open("/scr3/jruffio/data/kpic/models/planets_templates/lte018-5.0-0.0a+0.0.BT-Settl.spec.7", 'r') as f:
                        model_wvs = []
                        model_fluxes = []
                        for line in f.readlines():
                            line_args = line.strip().split()
                            model_wvs.append(float(line_args[0]))
                            model_fluxes.append(float(line_args[1].replace('D', 'E')))
                    model_wvs = np.array(model_wvs)/1.e4
                    model_fluxes = np.array(model_fluxes)
                    model_fluxes = 10 ** (model_fluxes - 8)
                    crop_plmodel = np.where((model_wvs>1.8-(2.6-1.8)/2)*(model_wvs<2.6+(2.6-1.8)/2))
                    model_wvs = model_wvs[crop_plmodel]
                    model_fluxes = model_fluxes[crop_plmodel]
                    # print(model_wvs)
                    # plt.plot(model_wvs,model_fluxes)
                    # plt.show()
                    # planet_convspec = convolve_spectrum(wmod, ori_planet_spec, 30000, specpool)
                    if 1:
                        pl_line_widths = np.array(pd.DataFrame(line_width_func(model_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                        pl_pixel_widths = np.array(pd.DataFrame(pixel_width_func(model_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                        planet_convspec = convolve_spectrum_line_width(model_wvs,model_fluxes,pl_line_widths,mypool=specpool)
                        planet_convspec = convolve_spectrum_pixel_width(model_wvs,planet_convspec,pl_pixel_widths,mypool=specpool)
                    planet_convspec /= np.nanmean(planet_convspec)
                    science_model_spline = interpolate.splrep(model_wvs, planet_convspec)

            phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
            with pyfits.open(phoenix_wv_filename) as hdulist:
                phoenix_wvs = hdulist[0].data/1.e4
            crop_phoenix = np.where((phoenix_wvs>1.8-(2.6-1.8)/2)*(phoenix_wvs<2.6+(2.6-1.8)/2))
            phoenix_wvs = phoenix_wvs[crop_phoenix]
            if 1: # A0 star model
                with pyfits.open(phoenix_A0_filename) as hdulist:
                    phoenix_A0 = hdulist[0].data[crop_phoenix]
                print("convolving: "+phoenix_A0_filename)
                phoenix_line_widths = np.array(pd.DataFrame(line_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                phoenix_A0_conv = convolve_spectrum_line_width(phoenix_wvs,phoenix_A0,phoenix_line_widths,mypool=specpool)
                phoenix_A0_conv = convolve_spectrum_pixel_width(phoenix_wvs,phoenix_A0_conv,phoenix_pixel_widths,mypool=specpool)
                phoenix_A0_conv /= np.nanmean(phoenix_A0_conv)
                # phoenix_A0_spline = interpolate.splrep(phoenix_wvs, phoenix_A0_conv)
                phoenix_A0_func = interp1d(phoenix_wvs, phoenix_A0_conv, bounds_error=False, fill_value=np.nan)
            if phoenix_host_filename == phoenix_A0_filename:  # host star model
                phoenix_host_func = phoenix_A0_func
            else:
                with pyfits.open(phoenix_host_filename) as hdulist:
                    phoenix_host = hdulist[0].data[crop_phoenix]
                print("convolving: "+phoenix_host_filename)
                phoenix_line_widths = np.array(pd.DataFrame(line_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                phoenix_host_conv = convolve_spectrum_line_width(phoenix_wvs,phoenix_host,phoenix_line_widths,mypool=specpool)
                phoenix_host_conv = convolve_spectrum_pixel_width(phoenix_wvs,phoenix_host_conv,phoenix_pixel_widths,mypool=specpool)
                phoenix_host_conv /= np.nanmean(phoenix_host_conv)
                # phoenix_host_spline = interpolate.splrep(phoenix_wvs, phoenix_host_conv)
                phoenix_host_func = interp1d(phoenix_wvs, phoenix_host_conv, bounds_error=False, fill_value=np.nan)

            print("Done convolving: "+phoenix_host_filename)
            if 1:
                science_spec_hpf = np.zeros(science_spec.shape)
                science_spec_lpf = np.zeros(science_err.shape)
                sciencehost_spec_hpf = np.zeros(sciencehost_spec.shape)
                sciencehost_spec_lpf = np.zeros(sciencehost_err.shape)
                dark1_spec_hpf = np.zeros(science_spec.shape)
                dark2_spec_hpf = np.zeros(science_spec.shape)
                slit1_spec_hpf = np.zeros(science_spec.shape)
                for order_id in range(Norders):
                    p = science_spec[order_id,:]
                    p_lpf, p_hpf = LPFvsHPF(p, cutoff=cutoff)
                    science_spec_lpf[order_id,:] = p_lpf
                    science_spec_hpf[order_id,:] = p_hpf

                    p = sciencehost_spec[order_id,:]
                    p_lpf, p_hpf = LPFvsHPF(p, cutoff=cutoff)
                    sciencehost_spec_lpf[order_id,:] = p_lpf
                    sciencehost_spec_hpf[order_id,:] = p_hpf

                    _, dark1_spec_hpf[order_id,:] = LPFvsHPF(dark1_spec[order_id,:], cutoff=cutoff)
                    _, dark2_spec_hpf[order_id,:] = LPFvsHPF(dark2_spec[order_id,:], cutoff=cutoff)
                    _, slit1_spec_hpf[order_id,:] = LPFvsHPF(slit1_spec[order_id,:], cutoff=cutoff)
                    # dark1_spec_hpf[order_id,:] = dark1_spec[order_id,:]
                    # dark2_spec_hpf[order_id,:] = dark2_spec[order_id,:]
                    # slit1_spec_hpf[order_id,:] = slit1_spec[order_id,:]
                    # dark1_spec_hpf[order_id,:] = np.random.randn(np.size(dark1_spec[order_id,:]))*np.nanstd(dark1_spec[order_id,:])
                    # dark2_spec_hpf[order_id,:] = np.random.randn(np.size(dark1_spec[order_id,:]))*np.nanstd(dark2_spec[order_id,:])
                    # slit1_spec_hpf[order_id,:] = np.random.randn(np.size(dark1_spec[order_id,:]))*np.nanstd(slit1_spec[order_id,:])

                    # plt.plot(dark1_spec_hpf[order_id,:])
                    # plt.plot(dark2_spec_hpf[order_id,:])
                    # plt.plot(slit1_spec_hpf[order_id,:])
                    # plt.show()


                tmp_dwvs = wvs[:, 1:wvs.shape[-1]] - wvs[:, 0:wvs.shape[-1]-1]
                tmp_dwvs = np.concatenate([tmp_dwvs, tmp_dwvs[:, -1][:, None]], axis=1)
                min_dwv = np.min(tmp_dwvs)
                wvs4broadening = np.arange(np.min(wvs) - min_dwv * 150, np.max(wvs) + min_dwv * 150,
                                           min_dwv / 5)
                planet_convspec_broadsampling = interpolate.splev(wvs4broadening, science_model_spline, der=0)

                # plt.plot(wvs4broadening,planet_convspec_broadsampling)
                # plt.show()

                fluxout = np.zeros([2,6, np.size(vsini_list), np.size(rv_list)])
                logpostout = np.zeros([6, np.size(vsini_list), np.size(rv_list)])
                dAICout = np.zeros([6, np.size(vsini_list), np.size(rv_list)])

                # rv_list = np.arange(0,60,1)[:,None]

                print("Starting fit ")
                # a,c,b = _fitRV((vsini_list[0], wvs, science_spec_hpf, science_spec_lpf, sciencehost_spec_hpf, sciencehost_spec_lpf, science_err, slit1_spec_hpf, dark1_spec_hpf, dark2_spec_hpf, \
                # wvs4broadening, planet_convspec_broadsampling, A0_spec, phoenix_A0_func, phoenix_host_func, \
                # A0_rv, A0_baryrv, host_rv, science_baryrv, c_kms, cutoff, rv_list))
                # print(a)
                # print(b)
                # exit()
                outputs_list = specpool.map(_fitRV, zip(vsini_list,
                                                         itertools.repeat(wvs),
                                                         itertools.repeat(science_spec_hpf),
                                                         itertools.repeat(science_spec_lpf),
                                                         itertools.repeat(sciencehost_spec_hpf),
                                                         itertools.repeat(sciencehost_spec_lpf),
                                                         itertools.repeat(science_err),
                                                         itertools.repeat(slit1_spec_hpf),
                                                         itertools.repeat(dark1_spec_hpf),
                                                         itertools.repeat(dark2_spec_hpf),
                                                         itertools.repeat(wvs4broadening),
                                                         itertools.repeat(planet_convspec_broadsampling),
                                                         itertools.repeat(A0_spec),
                                                         itertools.repeat(phoenix_A0_func),
                                                         itertools.repeat(phoenix_host_func),
                                                         itertools.repeat(A0_rv),
                                                         itertools.repeat(A0_baryrv),
                                                         itertools.repeat(host_rv),
                                                         itertools.repeat(science_baryrv),
                                                         itertools.repeat(c_kms),
                                                         itertools.repeat(cutoff),
                                                         itertools.repeat(rv_list)))
                for vsini_id,out in enumerate(outputs_list):
                    vsini,_fluxout,_dAICout, _logpostout = out
                    print("returning",vsini,vsini_id)
                    fluxout[:,:,vsini_id,:] = _fluxout
                    dAICout[:,vsini_id,:] = _dAICout
                    logpostout[:,vsini_id,:] = _logpostout
                # print(fluxout)
                # exit()

            if save:
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=fluxout))
                hdulist.append(pyfits.ImageHDU(data=dAICout))
                hdulist.append(pyfits.ImageHDU(data=logpostout))
                hdulist.append(pyfits.ImageHDU(data=vsini_list))
                hdulist.append(pyfits.ImageHDU(data=rv_list))
                if combined:
                    if not os.path.exists(os.path.join(sciencedir, "out")):
                        os.makedirs(os.path.join(sciencedir, "out"))
                    out = os.path.join(sciencedir, "out","flux_and_posterior.fits")
                else:
                    if not os.path.exists(os.path.join(os.path.dirname(sciencefilename), "out")):
                        os.makedirs(os.path.join(os.path.dirname(sciencefilename), "out"))
                    out = os.path.join(os.path.dirname(sciencefilename), "out",os.path.basename(sciencefilename).replace(".fits","_flux_and_posterior.fits"))
                try:
                    hdulist.writeto(out, overwrite=True)
                except TypeError:
                    hdulist.writeto(out, clobber=True)
                hdulist.close()
                # exit()
        else:
            if combined:
                out = os.path.join(sciencedir, "out","flux_and_posterior.fits")
            else:
                raise(Exception())
                # out = os.path.join(os.path.dirname(sciencefilename), "out",os.path.basename(sciencefilename).replace(".fits","_flux_and_posterior.fits"))
            with pyfits.open(out) as hdulist:
                fluxout = hdulist[0].data
                dAICout = hdulist[1].data
                logpostout = hdulist[2].data
                vsini_list = hdulist[3].data
                rv_list = hdulist[4].data

        argmaxvsini,argmaxrv = np.unravel_index(np.argmax(logpostout[0,:,:]),logpostout[0,:,:].shape)
        _fluxout = fluxout[0,:,argmaxvsini,:]#/np.nanstd(fluxout[3::,argmaxvsini,:])
        _fluxout_err = fluxout[1,:,argmaxvsini,:]#/np.nanstd(fluxout[3::,argmaxvsini,:])
        legend_list = ["data",
                       "simulated (i.e., auto correl.)",
                       "background 1",
                       "dark 1",
                       "dark 2"]
        linestyle_list = ["-","-","--",":",":"]
        color_list = ["orange","blue","black","grey","grey"]
        plt.figure(1,figsize=(16,8))
        plt.subplot(2,2,1)
        for data_id,(name,ls,c) in enumerate(zip(legend_list,linestyle_list,color_list)):
            if data_id == 1:
                continue
            plt.fill_between(rv_list,fluxout[0,data_id,argmaxvsini,:]-fluxout[1,data_id,argmaxvsini,:],fluxout[0,data_id,argmaxvsini,:]+fluxout[1,data_id,argmaxvsini,:],color=c,alpha=0.5)
            plt.plot(rv_list,fluxout[0,data_id,argmaxvsini,:],alpha=1,label=name,linestyle=ls,color=c)
        plt.ylabel("Flux")
        plt.xlabel("rv (km/s)")
        plt.legend()
        print(argmaxvsini)
        print(fluxout[0,:,argmaxvsini,:].shape)
        print(np.size(np.where(np.abs(rv_list)>50)[0]))
        print(fluxout[0,:,argmaxvsini,np.where(np.abs(rv_list)>50)[0]].shape)
        print(np.nanmean(fluxout[0,:,argmaxvsini,np.where(np.abs(rv_list)>50)[0]],axis=1).shape)
        _fluxout = fluxout[0,:,argmaxvsini,:] - np.nanmean(fluxout[0,:,argmaxvsini,np.where(np.abs(rv_list)>50)[0]],axis=0)[:,None]
        print(_fluxout.shape)
        # _fluxout[2,:] = np.nan
        # _fluxout = _fluxout/np.nanstd(_fluxout[2,:])
        plt.subplot(2,2,2)
        for data_id,(name,ls,c) in enumerate(zip(legend_list,linestyle_list,color_list)):
            if data_id == 1:
                continue
            # plt.fill_between(rv_list,_fluxout[data_id,:]-fluxout[1,data_id,argmaxvsini,:],_fluxout[data_id,:]+fluxout[1,data_id,argmaxvsini,:],color=c,alpha=0.5)
            if data_id >= 2:
                sig = r" ($\sigma=${0:0.1f})".format(np.nanstd(_fluxout[data_id,:]/fluxout[1,data_id,argmaxvsini,:]))
            else:
                sig = ""
            plt.plot(rv_list,_fluxout[data_id,:]/fluxout[1,data_id,argmaxvsini,:],alpha=1,label=name+sig,linestyle=ls,color=c)
        plt.ylabel("SNR")
        plt.xlabel("rv (km/s)")
        # plt.ylim([-3,5])
        plt.legend()

        post = np.exp(logpostout[0,:,:] - np.nanmax(logpostout[0,:,:]))
        plt.subplot(2,2,3)
        rvpost = np.nansum(post,axis=0)
        plt.plot(rv_list,rvpost/np.nanmax(rvpost))
        plt.xlabel("RV (km/s)")
        plt.xlim([-50,50])
        plt.subplot(2,2,4)
        vsisipost = np.nansum(post,axis=1)
        plt.plot(vsini_list,vsisipost/np.nanmax(vsisipost))
        plt.xlabel("vsin(i) (km/s)")

        if 1:
            print("Saving " + out.replace(".fits",".png"))
            plt.savefig(out.replace(".fits",".png"))
            plt.savefig(out.replace(".fits",".pdf"))

        # plt.figure(3)
        # plt.subplot(1,2,1)
        # plt.imshow(post,interpolation="nearest",origin="lower")
        # plt.subplot(1,2,2)
        # plt.imshow(logpostout[0,:,:],interpolation="nearest",origin="lower")


        plt.show()
