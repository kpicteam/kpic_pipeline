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
from utils_2020.badpix import *
from utils_2020.misc import *
from utils_2020.spectra import *
from scipy import interpolate
from PyAstronomy import pyasl
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
    # plt.legend()
    # plt.show()
    return LPF_myvec,HPF_myvec


def _fitRV(paras):
    vsini,wvs,science_spec_hpf,science_spec_lpf, science_err,slit1_spec_hpf, dark1_spec_hpf, dark2_spec_hpf,\
    wvs4broadening,planet_convspec_broadsampling,A0_spec,phoenix_A0_func,\
    A0_rv,A0_baryrv,science_baryrv,c_kms,cutoff,rv_list = paras

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
    m1_norv = planet_broadspec_func(wvs) * transmission
    m1_norv = m1_norv / np.nanmean(m1_norv) * np.nanmean(science_spec_lpf)
    m1_norv_spec_hpf = np.zeros(science_spec_lpf.shape)
    for order_id in range(Norders):
        _, m1_norv_spec_hpf[order_id, :] = LPFvsHPF(m1_norv[order_id, :], cutoff=cutoff)

    fluxout = np.zeros((2,5,np.size(rv_list)))
    dAICout = np.zeros((5,np.size(rv_list)))
    logpostout = np.zeros((5,np.size(rv_list)))

    data_hpf_list = [science_spec_hpf, m1_norv_spec_hpf, slit1_spec_hpf, dark1_spec_hpf, dark2_spec_hpf]
    for data_id, data_hpf in enumerate(data_hpf_list):

        where_data_nans = np.where(np.isnan(data_hpf))
        transmission = A0_spec / phoenix_A0_func(wvs * (1 - (A0_rv - A0_baryrv) / c_kms))
        transmission[where_data_nans] = np.nan

        # m2 = phoenix_host_func(wvs * (1 - (host_rv - science_baryrv) / c_kms)) * transmission
        # for order_id in range(Norders):
        #     m2_tmp_lpf, m2_tmp_hpf = LPFvsHPF(m2[order_id, :], cutoff=cutoff)
        #     m2[order_id, :] = m2_tmp_hpf / m2_tmp_lpf * science_spec_lpf[order_id, :]

        for rv_id, rv in enumerate(rv_list):
            # print(vsini_id,np.size(vsini_list),rv_id,np.size(rv_list))
            m1 = planet_broadspec_func(wvs * (1 - (rv - science_baryrv) / c_kms)) * transmission
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

            where_data_finite = np.where(np.isfinite(ravelHPFdata))
            ravelHPFdata = ravelHPFdata[where_data_finite]
            ravelwvs = ravelwvs[where_data_finite]
            sigmas_vec = np.ravel(science_err)[where_data_finite]
            logdet_Sigma = np.sum(2 * np.log(sigmas_vec))

            m1_ravel = np.ravel(m1)[where_data_finite]
            # m2_ravel = np.ravel(m2)[where_data_finite]
            # HPFmodel_H0 = m2_ravel[:, None]
            # HPFmodel_H2 = m1_ravel[:, None]
            # HPFmodel = np.concatenate([m1_ravel[:, None], m2_ravel[:, None]], axis=1)
            # HPFmodel = m1_ravel[:,None]

            # print(np.where(np.isnan(sigmas_vec)))
            # print(np.sum(ravelHPFdata * m1_ravel / sigmas_vec ** 2) / np.sum((m1_ravel / sigmas_vec) ** 2))
            # print(np.sum(ravelHPFdata * m1_ravel / sigmas_vec ** 2) , np.sum((m1_ravel / sigmas_vec) ** 2))
            # print(np.sum(ravelHPFdata) , np.sum((m1_ravel) ** 2))
            # exit()

            # # print(np.where(np.isnan( ravelHPFdata/sigmas_vec)))
            # # print(np.where(np.isnan( HPFmodel/sigmas_vec[:,None])))
            # plt.plot(HPFmodel[:,0]/np.std(HPFmodel[:,0]),color="blue")
            # # plt.plot(HPFmodel[:,1]/np.std(HPFmodel[:,1]),color="red")
            # plt.plot(ravelHPFdata/np.std(ravelHPFdata),color="purple")
            # plt.plot(HPFmodel_H2/np.std(HPFmodel_H2),"--")
            # plt.show()

            if 1:
                # HPFparas = [np.sum(ravelHPFdata*HPFmodel_H2/sigmas_vec**2)/np.sum((HPFmodel_H2/sigmas_vec)**2),]
                # ravelHPFdata = ravelHPFdata / np.std(ravelHPFdata)
                # HPFmodel_H2 = HPFmodel_H2 / np.std(HPFmodel_H2)
                norm_HPFmodel = m1_ravel / sigmas_vec
                HPFparas = [np.sum(ravelHPFdata/ sigmas_vec * norm_HPFmodel ) / np.sum((norm_HPFmodel) ** 2), ]

                Npixs_HPFdata = np.size(ravelHPFdata)

                data_model = HPFparas[0] * m1_ravel
                ravelresiduals = ravelHPFdata - data_model
                HPFchi2 = np.nansum((ravelresiduals/sigmas_vec) ** 2)
                HPFchi2_H0 = np.nansum((ravelHPFdata/sigmas_vec) ** 2)

                a = np.zeros(np.size(HPFparas))
                a[0] = 1
                a_err = np.sqrt((HPFchi2 / Npixs_HPFdata)/np.sum(norm_HPFmodel**2))

                minus2logL_HPF = Npixs_HPFdata * (
                            1 + np.log(HPFchi2 / Npixs_HPFdata) + logdet_Sigma + np.log(2 * np.pi))
                minus2logL_HPF_H0 = Npixs_HPFdata * (
                            1 + np.log(HPFchi2_H0 / Npixs_HPFdata) + logdet_Sigma + np.log(2 * np.pi))
                AIC_HPF = 2 * 1 + minus2logL_HPF
                AIC_HPF_H0 = minus2logL_HPF_H0


                # covphi = HPFchi2 / Npixs_HPFdata * (1 / np.sum(m1_ravel ** 2))
                slogdet_icovphi0 = np.log(1 / np.sum((norm_HPFmodel) ** 2))

                fluxout[0,data_id, rv_id] = HPFparas[0]
                fluxout[1,data_id, rv_id] = a_err
                dAICout[data_id, rv_id] = AIC_HPF_H0-AIC_HPF
                logpostout[data_id, rv_id] = -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0 - (
                            Npixs_HPFdata - 1 + 2 - 1) / (2) * np.log(HPFchi2)
                # print(HPFparas)
                # print(rv_list)
                # print(vsini)
                # plt.plot(ravelHPFdata,label="data")
                # plt.plot(data_model,label="m") #HPFparas[0]*
                # plt.plot(ravelresiduals,label="residuals")
                # # tmp = np.ravel(LPFvsHPF(transmission[0, :], cutoff=cutoff)[1])[where_data_finite]
                # # plt.plot(tmp/np.nanstd(tmp)*np.nanstd(ravelresiduals),label="transmission")
                # plt.legend()
                # plt.show()
    return fluxout,dAICout,logpostout

if __name__ == "__main__":
    try:
        import mkl

        mkl.set_num_threads(1)
    except:
        pass


    mykpicdir = "/scr3/jruffio/data/kpic/"
    phoenix_folder = "/scr3/jruffio/data/kpic/models/phoenix/"



    fib = 1
    numthreads = 32
    # Combining oders 2M0746A 9 / np.sqrt(np.sum(1 / np.array([7, 8, 6, 7, 4, 4, 13.5, 7, 6]) ** 2))
    # selec_orders = [6]
    selec_orders = [5,6,7,8]
    # selec_orders = [0,1,2,5,6,7,8]
    # selec_orders = [0,1,2,3,4,5,6,7,8]
    # selec_orders = [1,2,6,7,8]
    Norders=len(selec_orders)
    cutoff = 5
    c_kms = 299792.458
    vsini_list = np.linspace(0,100,32,endpoint=True)
    # vsini_list = np.array([0,1])
    rv_list = np.concatenate([np.arange(-400, -20, 5), np.arange(-20,20, 0.1), np.arange(20, 400, 5)], axis=0)
    # rv_list = np.concatenate([np.arange(-400, -55.567-3, 5), np.arange(-3-55.567, 3-55.567, 0.01), np.arange(3-55.567, 400, 5)], axis=0)
    # rv_list = np.concatenate([np.arange(-400, -85.391-3, 5), np.arange(-3-85.391, 3-85.391, 0.01), np.arange(3-85.391, 400, 5)], axis=0)
    # rv_list = np.concatenate([np.arange(-400, 55-10, 5), np.arange(55-10, 55+10, 0.1), np.arange(55+10, 400, 5)], axis=0)
    # rv_list = np.concatenate([np.arange(-400, -10, 5), np.arange(-10, 10, 0.5), np.arange(10, 400, 5)], axis=0)
    # rv_list = np.linspace(-2,30,1,endpoint=True)
    # rv_list = np.linspace(-300,300,101,endpoint=True)
    save = True
    plotonly = False
    if 1:
        ## standard star selection
        # A0dir = os.path.join(mykpicdir,"20191012_HD_1160_A")
        # A0dir = os.path.join(mykpicdir,"20191108_HD_1160")
        # A0_rv = 12.6 #km/s
        # A0dir = os.path.join(mykpicdir,"20191013A_kap_And")
        # A0dir = os.path.join(mykpicdir,"20191013B_kap_And")
        # A0dir = os.path.join(mykpicdir,"20191107_kap_And")
        # A0dir = os.path.join(mykpicdir,"20191215_kap_And")
        # A0_rv = -12.7 #km/s
        # A0dir = os.path.join(mykpicdir,"20191014_HR_8799")
        # A0_rv = -12.6 #km/s
        # A0dir = os.path.join(mykpicdir,"20200607_ups_Her")
        # A0dir = os.path.join(mykpicdir,"20200609_ups_Her")
        # A0_rv = 3 #km/s
        # A0dir = os.path.join(mykpicdir,"20200608_d_Sco")
        # A0dir = os.path.join(mykpicdir,"20200609_d_Sco")
        # A0dir = os.path.join(mykpicdir,"20200702_d_Sco")
        A0dir = os.path.join(mykpicdir,"20200703_d_Sco")
        A0_rv = -13 #km/s
        # A0dir = os.path.join(mykpicdir,"20200702_15_Sgr")
        # A0_rv = -6 #km/s
        # A0dir = os.path.join(mykpicdir,"20200607_5_Vul")
        # A0_rv = -21.2 #km/s
        # A0dir = os.path.join(mykpicdir,"20200701_15_Sgr")
        # A0_rv = -6 #km/s
        phoenix_A0_filename = glob(os.path.join(phoenix_folder, "kap_And" + "*.fits"))[0]
        filelist = glob(os.path.join(A0dir, "*fluxes.fits"))
        A0_spec,A0_err,_,_,A0_baryrv = combine_spectra_from_folder(filelist,"star")
        cp_A0_spec = copy(edges2nans(A0_spec))
        cp_A0_err = copy(edges2nans(A0_err))
    if 1:
        ## science selection
        # sciencedir = os.path.join(mykpicdir,"20191012_2M0746A")
        # sciencedir = os.path.join(mykpicdir,"20191012_2M0746B")
        # sciencedir = os.path.join(mykpicdir,"20191013B_gg_Tau")
        # sciencedir = os.path.join(mykpicdir,"20191013B_gg_Tau_B")
        # sciencedir = os.path.join(mykpicdir,"20191014_HIP_12787_A")
        # sciencedir = os.path.join(mykpicdir,"20191014_HIP_12787_B")
        # sciencedir = os.path.join(mykpicdir,"20191108_DH_Tau_B")
        # sciencedir = os.path.join(mykpicdir,"20191108_DH_Tau")
        # sciencedir = os.path.join(mykpicdir, "20191107_kap_And_B")
        # sciencedir = os.path.join(mykpicdir, "20191013B_kap_And_B")
        # sciencedir = os.path.join(mykpicdir,"20191108_2M0746A")
        # sciencedir = os.path.join(mykpicdir,"20191108_2M0746B")
        # sciencedir = os.path.join(mykpicdir,"20191108_HIP_12787_A")
        # sciencedir = os.path.join(mykpicdir,"20191108_HIP_12787_B")
        # sciencedir = os.path.join(mykpicdir,"20191215_kap_And_B")
        # sciencedir = os.path.join(mykpicdir,"20191215_DH_Tau_B")
        # sciencedir = os.path.join(mykpicdir,"20191215_DH_Tau")
        # sciencedir = os.path.join(mykpicdir,"20200607_HIP_81497")
        # sciencedir = os.path.join(mykpicdir,"20200607_HIP_95771")
        # sciencedir = os.path.join(mykpicdir,"20200608_HIP_81497_30s")
        # sciencedir = os.path.join(mykpicdir,"20200608_HIP_81497_7.5s")
        # sciencedir = os.path.join(mykpicdir,"20200609_HIP_81497")
        # sciencedir = os.path.join(mykpicdir,"20200609_ROXs_12B")
        # sciencedir = os.path.join(mykpicdir,"20200701_ROXs_42Bb")
        # sciencedir = os.path.join(mykpicdir,"20200702_ROXs_42Bb")
        # sciencedir = os.path.join(mykpicdir,"20200702_ROXs_42B")
        # sciencedir = os.path.join(mykpicdir,"20200702_ROXs_42B_daytime") #[ 15.93993899 305.19731602  15.53791296  14.55284422]
        # sciencedir = os.path.join(mykpicdir,"20200702_ROXs_42B_fit") #[ 14.45913059 304.17032175  14.0525108   13.23600295]
        # sciencedir = os.path.join(mykpicdir,"20200702_ROXs_42B_chopping") #[ 71.24556938 316.18770182 167.29706758  59.05578673]
        sciencedir = os.path.join(mykpicdir,"20200703_ROXs_12B")
        # sciencedir = os.path.join(mykpicdir,"20200703_ROXs_12B_chopping")
        filelist = glob(os.path.join(sciencedir, "*fluxes.fits"))
        filelist.sort()
        # filelist = [filelist[0],filelist[2],filelist[4],filelist[6],filelist[8]]
        # filelist = [filelist[1],filelist[3],filelist[5],filelist[7],filelist[9]]
        print(filelist)
        print("len(filelist)",len(filelist))
        # exit()
    combined = True
    # combined = False
    if not plotonly:
        if 1:
            science_spec,science_err,slit_spec,dark_spec,science_baryrv = combine_spectra_from_folder(filelist,"science")
            # science_spec,science_err,slit_spec,dark_spec,science_baryrv = combine_spectra_from_folder(filelist,"star")
            # print(np.nansum(science_spec,axis=(1,2)))
            # exit()
        # for sciencefilename in filelist[::-1]:
        #     print(sciencefilename)
        #     science_spec,science_err,slit_spec,dark_spec,science_baryrv = combine_spectra_from_folder([sciencefilename],"science")
        #     print(np.nanmean(science_spec,axis=(1,2)))
        #     if  "gg_Tau" in os.path.basename(sciencedir) or \
        #             "HIP_81497" in os.path.basename(sciencedir) or \
        #             "HIP_95771" in os.path.basename(sciencedir) or \
        #             "HIP_12787" in os.path.basename(sciencedir) or \
        #             ("DH_Tau" in os.path.basename(sciencedir) and not "DH_Tau_B" in os.path.basename(sciencedir)):
        #         fib = np.nanargmax(np.nanmean(science_spec,axis=(1,2)))

            A0_spec,A0_err = cp_A0_spec[fib,selec_orders],cp_A0_err[fib,selec_orders]
            # host_spec,host_err = cp_host_spec[fib,selec_orders],cp_host_err[fib,selec_orders]
            science_spec = edges2nans(science_spec)
            science_err = edges2nans(science_err)
            science_spec, science_err = science_spec[fib,selec_orders],science_err[fib,selec_orders]
            slit1_spec, dark1_spec = slit_spec[fib,selec_orders],dark_spec[fib,selec_orders]
            dark2_spec = dark_spec[fib+3,selec_orders]
            where_nans = np.where(np.isnan(A0_spec)+np.isnan(science_spec))
            science_spec[where_nans] = np.nan
            science_err[where_nans] = np.nan
            dark1_spec[where_nans] = np.nan
            dark2_spec[where_nans] = np.nan
            slit1_spec[where_nans] = np.nan

            hdulist = pyfits.open(glob(os.path.join(sciencedir, "calib", "*_wvs.fits"))[0])
            wvs = hdulist[0].data[fib,selec_orders,:]
            #
            # print(wvs.shape,science_spec.shape)
            # for k in range(Norders):
            #     plt.subplot(Norders,1,Norders-k)
            #     plt.fill_between(wvs[k,:], science_spec[k,:]-science_err[k,:], science_spec[k,:]+science_err[k,:],label="Error bars",color="orange")
            #     plt.plot(wvs[k,:], science_spec[k,:],label="spec",color="blue",linewidth=0.5)
            #     # plt.fill_between(wvs[k,:], host_spec[k,:]-host_err[k,:], host_spec[k,:]+host_err[k,:],label="Error bars",color="orange")
            #     # plt.plot(wvs[k,:], host_spec[k,:],label="spec",color="blue",linewidth=0.5)
            #     # plt.plot(wvs[k,:], slit1_spec[k,:],label="slit background 1",alpha=0.5,color="grey")
            #     # plt.plot(wvs[k,:], dark1_spec[k,:],label="dark background 1",alpha=0.5,color="grey")
            #     # plt.plot(wvs[k,:], slit2_spec[k,:],label="slit background 2",alpha=0.5,linestyle="--",color="grey")
            #     # plt.plot(wvs[k,:], dark2_spec[k,:],label="dark background 2",alpha=0.5,linestyle="--",color="grey")
            #     plt.ylim([0,120])
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
            if "gg_Tau" in sciencedir or "HIP_12787" in sciencedir or ("DH_Tau" in sciencedir and not ("DH_Tau_B" in sciencedir)) or ("ROXs_42B" in sciencedir and not ("ROXs_42Bb" in sciencedir)): #
                phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
                with pyfits.open(phoenix_wv_filename) as hdulist:
                    phoenix_wvs = hdulist[0].data/1.e4
                wvs_max = np.nanmax(wvs)
                wvs_min = np.nanmin(wvs)
                crop_phoenix = np.where((phoenix_wvs>wvs_min-(wvs_max-wvs_min))*(phoenix_wvs<wvs_max+(wvs_max-wvs_min)))
                phoenix_wvs = phoenix_wvs[crop_phoenix]

                phoenix_science_filename = glob(os.path.join(phoenix_folder, "DH_Tau_" + "*.fits"))[0]

                with pyfits.open(phoenix_science_filename) as hdulist:
                    phoenix_science = hdulist[0].data[crop_phoenix]
                print("convolving: "+phoenix_science_filename)
                if 1:
                    phoenix_line_widths = np.array(pd.DataFrame(line_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                    phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                    phoenix_science_conv = convolve_spectrum_line_width(phoenix_wvs,phoenix_science,phoenix_line_widths,mypool=specpool)
                    phoenix_science_conv = convolve_spectrum_pixel_width(phoenix_wvs,phoenix_science_conv,phoenix_pixel_widths,mypool=specpool)
                phoenix_science_conv /= np.nanmean(phoenix_science_conv)
                # phoenix_science_spline = interpolate.splrep(phoenix_wvs, phoenix_science_conv)
                science_model_spline = interpolate.splrep(phoenix_wvs, phoenix_science_conv)
            if "HIP_81497" in sciencedir or "HIP_95771" in sciencedir:
                phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
                with pyfits.open(phoenix_wv_filename) as hdulist:
                    phoenix_wvs = hdulist[0].data/1.e4
                wvs_max = np.nanmax(wvs)
                wvs_min = np.nanmin(wvs)
                crop_phoenix = np.where((phoenix_wvs>wvs_min-(wvs_max-wvs_min))*(phoenix_wvs<wvs_max+(wvs_max-wvs_min)))
                phoenix_wvs = phoenix_wvs[crop_phoenix]

                phoenix_science_filename = glob(os.path.join(phoenix_folder, "HIP_81497_" + "*.fits"))[0]

                with pyfits.open(phoenix_science_filename) as hdulist:
                    phoenix_science = hdulist[0].data[crop_phoenix]
                print("convolving: "+phoenix_science_filename)
                if 1:
                    phoenix_line_widths = np.array(pd.DataFrame(line_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                    phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                    phoenix_science_conv = convolve_spectrum_line_width(phoenix_wvs,phoenix_science,phoenix_line_widths,mypool=specpool)
                    phoenix_science_conv = convolve_spectrum_pixel_width(phoenix_wvs,phoenix_science_conv,phoenix_pixel_widths,mypool=specpool)
                phoenix_science_conv /= np.nanmean(phoenix_science_conv)
                # phoenix_science_spline = interpolate.splrep(phoenix_wvs, phoenix_science_conv)
                science_model_spline = interpolate.splrep(phoenix_wvs, phoenix_science_conv)
            # if "ROXs_12B" in sciencedir:
            #     phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
            #     with pyfits.open(phoenix_wv_filename) as hdulist:
            #         phoenix_wvs = hdulist[0].data/1.e4
            #     wvs_max = np.nanmax(wvs)
            #     wvs_min = np.nanmin(wvs)
            #     crop_phoenix = np.where((phoenix_wvs>wvs_min-(wvs_max-wvs_min))*(phoenix_wvs<wvs_max+(wvs_max-wvs_min)))
            #     phoenix_wvs = phoenix_wvs[crop_phoenix]
            #
            #     phoenix_science_filename = glob(os.path.join(phoenix_folder, "ROXs_12B" + "*.fits"))[0]
            #
            #     with pyfits.open(phoenix_science_filename) as hdulist:
            #         phoenix_science = hdulist[0].data[crop_phoenix]
            #     print("convolving: "+phoenix_science_filename)
            #     if 1:
            #         phoenix_line_widths = np.array(pd.DataFrame(line_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
            #         phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
            #         phoenix_science_conv = convolve_spectrum_line_width(phoenix_wvs,phoenix_science,phoenix_line_widths,mypool=specpool)
            #         phoenix_science_conv = convolve_spectrum_pixel_width(phoenix_wvs,phoenix_science_conv,phoenix_pixel_widths,mypool=specpool)
            #     phoenix_science_conv /= np.nanmean(phoenix_science_conv)
            #     # phoenix_science_spline = interpolate.splrep(phoenix_wvs, phoenix_science_conv)
            #     science_model_spline = interpolate.splrep(phoenix_wvs, phoenix_science_conv)

            # if 0:
            #     planet_spec_func = interp1d(phoenix_wvs, phoenix_science_conv, bounds_error=False, fill_value=np.nan)
            #     molecule = "CO"
            #     osiris_data_dir = "/scr3/jruffio/data/kpic/models/"
            #     molecular_template_folder = os.path.join(osiris_data_dir, "molecular_templates")
            #     travis_mol_filename = os.path.join(molecular_template_folder,
            #                                        "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um." + molecule + "only.7")
            #     travis_mol_filename_D2E = os.path.join(molecular_template_folder,
            #                                            "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um." + molecule + "only.7_D2E")
            #     data = np.loadtxt(travis_mol_filename_D2E)
            #     print(data.shape)
            #     wmod = data[:, 0] / 10000.
            #     wmod_argsort = np.argsort(wmod)
            #     wmod = wmod[wmod_argsort]
            #     mol_temp = data[wmod_argsort, 1]
            #     planet_spec_func2 = interp1d(wmod, mol_temp, bounds_error=False, fill_value=np.nan)
            #
            #     print(wvs.shape, science_spec.shape)
            #     for k in range(Norders):
            #         plt.subplot(Norders, 1, Norders - k)
            #         _, tmp_hpf = LPFvsHPF(science_spec[k, :], cutoff=cutoff)
            #         _, tmp_model_hpf = LPFvsHPF(planet_spec_func(wvs[k, :] * (1 - (14.3 - science_baryrv) / c_kms)),
            #                                     cutoff=cutoff)
            #         _, tmp_model2_hpf = LPFvsHPF(planet_spec_func2(wvs[k, :] * (1 - (14.3 - science_baryrv) / c_kms)),
            #                                     cutoff=cutoff)
            #         # plt.fill_between(wvs[k,:], tmp_hpf-science_err[k,:],tmp_hpf+science_err[k,:],label="Error bars",color="orange")
            #         plt.plot(wvs[k, :], tmp_hpf, label="HPF spec", color="blue", linewidth=0.5)
            #         plt.plot(wvs[k, :], tmp_model_hpf / np.nanstd(tmp_model_hpf) * np.nanstd(tmp_hpf),
            #                  label="HPF M0 star", color="black", linewidth=1, linestyle="--")
            #         plt.plot(wvs[k, :], tmp_model2_hpf / np.nanstd(tmp_model2_hpf) * np.nanstd(tmp_hpf),
            #                  label="HPF CO model", color="black", linewidth=1, linestyle="-")
            #         plt.ylim([np.nanmin(tmp_hpf) - 5 * np.nanmedian(science_err[k, :]),
            #                   np.nanmax(tmp_hpf) + 5 * np.nanmedian(science_err[k, :])])
            #     plt.legend()
            #     plt.show()

                # exit()
            if "ROXs_12B" in sciencedir or "ROXs_42Bb" in sciencedir or "kap_And_B" in sciencedir:
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
                travis_spec_filename = os.path.join(sciencedir, "calib",
                                                    "lte2048-3.77-0.11.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019_COscl=1.00_H2Oscl=1.00_CH4scl=1.0_4KPIC.7_D2E_Ksorted_conv")
                if 0 or len(glob(travis_spec_filename)) == 0:
                    # /scr3/jruffio/data/kpic/models/planets_templates/lte2048-3.7_D2E7-0.11.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019_COscl=1.00_H2Oscl=1.00_CH4scl=1.0_4KPIC.7_D2E
                    # /scr3/jruffio/data/kpic/models/planets_templates/lte2048-3.77-0.11.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019_COscl=1.00_H2Oscl=1.00_CH4scl=1.0_4KPIC.7_D2E
                    data = np.loadtxt(os.path.join("/scr3/jruffio/data/kpic/models/planets_templates/",
                                                   "lte2048-3.77-0.11.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019_COscl=1.00_H2Oscl=1.00_CH4scl=1.0_4KPIC.7_D2E"))
                    print(data.shape)
                    wmod = data[:, 0] / 10000.
                    wmod_argsort = np.argsort(wmod)
                    wmod = wmod[wmod_argsort]
                    crop_moltemp = np.where((wmod > 1.8 - (2.6 - 1.8) / 2) * (wmod < 2.6 + (2.6 - 1.8) / 2))
                    # crop_moltemp = np.where(
                    #     (wmod > wvs[0,0] - (wvs[0,-1] - wvs[0,0]) / 2) * (wmod < wvs[-1,-1] + (wvs[-1,-1] - wvs[-1,0]) / 2))
                    wmod = wmod[crop_moltemp]
                    mol_temp = data[wmod_argsort, 1][crop_moltemp]
                    mol_temp = 10 ** (mol_temp - np.max(mol_temp))

                    if 1:
                        pl_line_widths = np.array(
                            pd.DataFrame(line_width_func(wmod)).interpolate(method="linear").fillna(
                                method="bfill").fillna(method="ffill"))[:, 0]
                        pl_pixel_widths = np.array(
                            pd.DataFrame(pixel_width_func(wmod)).interpolate(method="linear").fillna(
                                method="bfill").fillna(method="ffill"))[:, 0]
                        planet_convspec = convolve_spectrum_line_width(wmod, mol_temp, pl_line_widths,
                                                                       mypool=specpool)
                        planet_convspec = convolve_spectrum_pixel_width(wmod, planet_convspec, pl_pixel_widths,
                                                                        mypool=specpool)
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
                    # plt.plot(wmod, planet_convspec)
                    # print("coucou")
                    # plt.show()
                #
                with open(travis_spec_filename, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=' ')
                    list_starspec = list(csv_reader)
                    oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                    col_names = oriplanet_spec_str_arr[0]
                    ori_planet_spec = oriplanet_spec_str_arr[1::3, 1].astype(np.float)
                    wmod = oriplanet_spec_str_arr[1::3, 0].astype(np.float)
                    ori_planet_spec = 10 ** (ori_planet_spec - np.max(ori_planet_spec))

                    crop_plmodel = np.where((wmod > 1.8 - (2.6 - 1.8) / 2) * (wmod < 2.6 + (2.6 - 1.8) / 2))
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

                # interpolate.splev(wvs4broadening, science_model_spline, der=0)
                # filter = np.loadtxt("/scr3/jruffio/data/kpic/models/filters/Generic_Johnson_UBVRIJHKL.K.dat")
                # filter_func = interp1d(filter[:,0]/1e4,filter[:,1],bounds_error=False,fill_value=0)
                # dwmod = wmod[1::]-wmod[0:np.size(wmod)-1]
                # ori_planet_spec = ori_planet_spec/np.sum(ori_planet_spec[1::]*filter_func(wmod[1::]*dwmod))
                science_model_spline = interpolate.splrep(wmod, ori_planet_spec)
                # plt.plot(wmod, ori_planet_spec)
                # m = interpolate.splev(np.ravel(wvs), science_model_spline, der=0)
                # m /= np.nanmean(m)
                # print(np.sum((m-1)**2))
                # plt.plot(np.ravel(wvs), m)
                # plt.show()
                # exit()
            if "DH_Tau_B" in sciencedir or "2M0746" in sciencedir: #"ROXs_12B" in sciencedir or
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
                if 1:
                    phoenix_line_widths = np.array(pd.DataFrame(line_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                    phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                    phoenix_A0_conv = convolve_spectrum_line_width(phoenix_wvs,phoenix_A0,phoenix_line_widths,mypool=specpool)
                    phoenix_A0_conv = convolve_spectrum_pixel_width(phoenix_wvs,phoenix_A0_conv,phoenix_pixel_widths,mypool=specpool)
                phoenix_A0_conv /= np.nanmean(phoenix_A0_conv)
                # phoenix_A0_spline = interpolate.splrep(phoenix_wvs, phoenix_A0_conv)
                phoenix_A0_func = interp1d(phoenix_wvs, phoenix_A0_conv, bounds_error=False, fill_value=np.nan)
            # if 1: # host star model
            #     with pyfits.open(phoenix_host_filename) as hdulist:
            #         phoenix_host = hdulist[0].data[crop_phoenix]
            #     print("convolving: "+phoenix_host_filename)
            #     if 1:
            #         phoenix_line_widths = np.array(pd.DataFrame(line_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
            #         phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
            #         phoenix_host_conv = convolve_spectrum_line_width(phoenix_wvs,phoenix_host,phoenix_line_widths,mypool=specpool)
            #         phoenix_host_conv = convolve_spectrum_pixel_width(phoenix_wvs,phoenix_host_conv,phoenix_pixel_widths,mypool=specpool)
            #     phoenix_host_conv /= np.nanmean(phoenix_host_conv)
            #     # phoenix_host_spline = interpolate.splrep(phoenix_wvs, phoenix_host_conv)
            #     phoenix_host_func = interp1d(phoenix_wvs, phoenix_host_conv, bounds_error=False, fill_value=np.nan)


            if 1:
                science_spec_hpf = np.zeros(science_spec.shape)
                science_spec_lpf = np.zeros(science_err.shape)
                dark1_spec_hpf = np.zeros(science_spec.shape)
                dark2_spec_hpf = np.zeros(science_spec.shape)
                slit1_spec_hpf = np.zeros(science_spec.shape)
                slit2_spec_hpf = np.zeros(science_spec.shape)
                for order_id in range(Norders):
                    p = science_spec[order_id,:]
                    p_lpf, p_hpf = LPFvsHPF(p, cutoff=cutoff)
                    science_spec_lpf[order_id,:] = p_lpf
                    science_spec_hpf[order_id,:] = p_hpf

                    # plt.plot(p_lpf,label="p_lpf")
                    # plt.plot(p_hpf,label="p_hpf")
                    # plt.plot(p,label="p")
                    # plt.show()


                    _, dark1_spec_hpf[order_id,:] = LPFvsHPF(dark1_spec[order_id,:], cutoff=cutoff)
                    _, dark2_spec_hpf[order_id,:] = LPFvsHPF(dark2_spec[order_id,:], cutoff=cutoff)
                    _, slit1_spec_hpf[order_id,:] = LPFvsHPF(slit1_spec[order_id,:], cutoff=cutoff)


                tmp_dwvs = wvs[:, 1:wvs.shape[-1]] - wvs[:, 0:wvs.shape[-1]-1]
                tmp_dwvs = np.concatenate([tmp_dwvs, tmp_dwvs[:, -1][:, None]], axis=1)
                min_dwv = np.min(tmp_dwvs)
                wvs4broadening = np.arange(np.min(wvs) - min_dwv * 150, np.max(wvs) + min_dwv * 150,
                                           min_dwv / 5)
                planet_convspec_broadsampling = interpolate.splev(wvs4broadening, science_model_spline, der=0)

                # plt.plot(wvs4broadening,planet_convspec_broadsampling)
                # plt.show()

                fluxout = np.zeros([2,5, np.size(vsini_list), np.size(rv_list)])
                dAICout = np.zeros([5, np.size(vsini_list), np.size(rv_list)])
                logpostout = np.zeros([5, np.size(vsini_list), np.size(rv_list)])

                # rv_list = np.arange(0,60,1)[:,None]

                if 0:
                    planet_broadspec_func = interp1d(wvs4broadening, planet_convspec_broadsampling, bounds_error=False,
                                                     fill_value=np.nan)
                    transmission = A0_spec / phoenix_A0_func(wvs * (1 - (A0_rv - A0_baryrv) / c_kms))
                    m1 = planet_broadspec_func(wvs * (1 - (rv_list[0] - science_baryrv) / c_kms)) * transmission
                    print(wvs.shape, science_spec.shape)
                    for k in range(Norders):
                        plt.subplot(Norders, 1, Norders - k)
                        # plt.fill_between(wvs[k, :], science_spec[k, :] - science_err[k, :],science_spec[k, :] + science_err[k, :], label="Error bars", color="orange")
                        plt.plot(wvs[k, :], science_spec[k, :], label="spec", color="blue", linewidth=0.5)
                        plt.plot(wvs[k, :], m1[k, :]/np.nanmean(m1[k, :])*np.nanmean(science_spec[k, :]), label="model*transmission", color="orange", linewidth=1)
                        plt.plot(wvs[k, :], A0_spec[k, :]/np.nanmean(A0_spec[k, :])*np.nanmean(science_spec[k, :]), label="A0", color="black", linewidth=0.5)
                        # plt.fill_between(wvs[k,:], host_spec[k,:]-host_err[k,:], host_spec[k,:]+host_err[k,:],label="Error bars",color="orange")
                        # plt.plot(wvs[k,:], host_spec[k,:],label="spec",color="blue",linewidth=0.5)
                        # plt.plot(wvs[k,:], slit1_spec[k,:],label="slit background 1",alpha=0.5,color="grey")
                        # plt.plot(wvs[k,:], dark1_spec[k,:],label="dark background 1",alpha=0.5,color="grey")
                        # plt.plot(wvs[k,:], slit2_spec[k,:],label="slit background 2",alpha=0.5,linestyle="--",color="grey")
                        # plt.plot(wvs[k,:], dark2_spec[k,:],label="dark background 2",alpha=0.5,linestyle="--",color="grey")
                        plt.ylim([0, 120])
                        # plt.ylim([0-10*np.nanstd(slit1_spec[k,:]),np.nanmax(science_spec[k,:])+5*np.nanmedian(science_err[k,:])])
                    plt.legend()
                    plt.show()

                # a,b,c = _fitRV((vsini_list[0], wvs, science_spec_hpf, science_spec_lpf, science_err, slit1_spec_hpf, dark1_spec_hpf, dark2_spec_hpf, \
                # wvs4broadening, planet_convspec_broadsampling, A0_spec, phoenix_A0_func, \
                # A0_rv, A0_baryrv, science_baryrv, c_kms, cutoff, rv_list))
                # print(a)
                # print(b)
                # print(c)
                # exit()
                outputs_list = specpool.map(_fitRV, zip(vsini_list,
                                                         itertools.repeat(wvs),
                                                         itertools.repeat(science_spec_hpf),
                                                         itertools.repeat(science_spec_lpf),
                                                         itertools.repeat(science_err),
                                                         itertools.repeat(slit1_spec_hpf),
                                                         itertools.repeat(dark1_spec_hpf),
                                                         itertools.repeat(dark2_spec_hpf),
                                                         itertools.repeat(wvs4broadening),
                                                         itertools.repeat(planet_convspec_broadsampling),
                                                         itertools.repeat(A0_spec),
                                                         itertools.repeat(phoenix_A0_func),
                                                         itertools.repeat(A0_rv),
                                                         itertools.repeat(A0_baryrv),
                                                         itertools.repeat(science_baryrv),
                                                         itertools.repeat(c_kms),
                                                         itertools.repeat(cutoff),
                                                         itertools.repeat(rv_list)))
                for vsini_id,out in enumerate(outputs_list):
                    _fluxout,_dAICout, _logpostout = out
                    fluxout[:,:,vsini_id,:] = _fluxout
                    dAICout[:,vsini_id,:] = _dAICout
                    logpostout[:,vsini_id,:] = _logpostout

            if save:
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=fluxout))
                hdulist.append(pyfits.ImageHDU(data=dAICout))
                hdulist.append(pyfits.ImageHDU(data=logpostout))
                hdulist.append(pyfits.ImageHDU(data=vsini_list))
                hdulist.append(pyfits.ImageHDU(data=rv_list))

                order_suffix = ""
                for myorder in selec_orders:
                    order_suffix += "_{0}".format(myorder)

                if combined:
                    if not os.path.exists(os.path.join(sciencedir, "out")):
                        os.makedirs(os.path.join(sciencedir, "out"))
                    out = os.path.join(sciencedir, "out", "flux_and_posterior"+order_suffix+".fits")
                else:
                    if not os.path.exists(os.path.join(os.path.dirname(sciencefilename), "out")):
                        os.makedirs(os.path.join(os.path.dirname(sciencefilename), "out"))
                    out = os.path.join(os.path.dirname(sciencefilename), "out",os.path.basename(sciencefilename).replace(".fits","_flux_and_posterior"+order_suffix+".fits"))
                try:
                    hdulist.writeto(out, overwrite=True)
                except TypeError:
                    hdulist.writeto(out, clobber=True)
                hdulist.close()
    else:
        order_suffix = ""
        for myorder in selec_orders:
            order_suffix += "_{0}".format(myorder)
        if combined:
            out = os.path.join(sciencedir, "out", "flux_and_posterior"+order_suffix+".fits")
        else:
            raise (Exception())
            # out = os.path.join(os.path.dirname(sciencefilename), "out",os.path.basename(sciencefilename).replace(".fits","_flux_and_posterior.fits"))

        with pyfits.open(out) as hdulist:
            fluxout = hdulist[0].data
            dAICout = hdulist[1].data
            logpostout = hdulist[2].data
            vsini_list = hdulist[3].data
            rv_list = hdulist[4].data

    _logpostout = logpostout[0,:,:]
    argmaxvsini, argmaxrv = np.unravel_index(np.argmax(_logpostout), _logpostout.shape)
    argmaxvsini = 0
    _fluxout = fluxout[0, :, argmaxvsini, :]  # /np.nanstd(fluxout[3::,argmaxvsini,:])
    _fluxout_err = fluxout[1, :, argmaxvsini, :]  # /np.nanstd(fluxout[3::,argmaxvsini,:])
    legend_list = ["data",
                   "simulated (i.e., auto correl.)",
                   "background 1",
                   "dark 1",
                   "dark 2"]
    linestyle_list = ["-", "-", "--", ":", ":"]
    color_list = ["orange", "blue", "black", "grey", "grey"]
    plt.figure(1, figsize=(16, 8))
    plt.subplot(2, 2, 1)
    for data_id, (name, ls, c) in enumerate(zip(legend_list, linestyle_list, color_list)):
        if data_id == 1:
            continue
        plt.fill_between(rv_list, _fluxout[data_id,:] -  _fluxout_err[data_id,:] ,
                         _fluxout[data_id, :] + _fluxout_err[data_id, :], color=c,
                         alpha=0.5)
        plt.plot(rv_list, fluxout[0, data_id, argmaxvsini, :], alpha=1, label=name, linestyle=ls, color=c)
    plt.ylabel("Flux")
    plt.xlabel("rv (km/s)")
    plt.legend()
    print(argmaxvsini)
    print(fluxout[0, :, argmaxvsini, :].shape)
    print(np.size(np.where(np.abs(rv_list) > 150)[0]))
    print(fluxout[0, :, argmaxvsini, np.where(np.abs(rv_list) > 150)[0]].shape)
    print(np.nanmean(fluxout[0, :, argmaxvsini, np.where(np.abs(rv_list) > 150)[0]], axis=1).shape)
    _fluxout = fluxout[0, :, argmaxvsini, :] - np.nanmean(
        fluxout[0, :, argmaxvsini, np.where(np.abs(rv_list) > 150)[0]], axis=0)[:, None]
    print(_fluxout.shape)
    # _fluxout[2,:] = np.nan
    # _fluxout = _fluxout/np.nanstd(_fluxout[2,:])
    plt.subplot(2, 2, 2)
    for data_id, (name, ls, c) in enumerate(zip(legend_list, linestyle_list, color_list)):
        if data_id == 1:
            continue
        # plt.fill_between(rv_list,_fluxout[data_id,:]-fluxout[1,data_id,argmaxvsini,:],_fluxout[data_id,:]+fluxout[1,data_id,argmaxvsini,:],color=c,alpha=0.5)
        if data_id >= 2:
            sig = r" ($\sigma=${0:0.1f})".format(
                np.nanstd(_fluxout[data_id, :] / fluxout[1, data_id, argmaxvsini, :]))
        else:
            sig = ""
        plt.plot(rv_list, _fluxout[data_id, :] / fluxout[1, data_id, argmaxvsini, :], alpha=1, label=name + sig,
                 linestyle=ls, color=c)
    plt.ylabel("SNR")
    plt.xlabel("rv (km/s)")
    # plt.ylim([-3,5])
    plt.legend()

    post = np.exp(logpostout[0, :, :] - np.nanmax(logpostout[0, :, :]))
    dvsini_list = vsini_list[1::]-vsini_list[0:np.size(vsini_list)-1]
    dvsini_list = np.insert(dvsini_list,0,[dvsini_list[0]])
    drv_list = rv_list[1::]-rv_list[0:np.size(rv_list)-1]
    drv_list = np.insert(drv_list,0,[drv_list[0]])
    print(np.size(dvsini_list),np.size(vsini_list))
    print(np.size(drv_list),np.size(rv_list))
    plt.subplot(2, 2, 3)
    rvpost = np.nansum(post*dvsini_list[:,None], axis=0)
    plt.plot(rv_list, rvpost / np.nanmax(rvpost))
    plt.xlabel("RV (km/s)")
    plt.xlim([-10,10])
    plt.subplot(2, 2, 4)
    vsisipost = np.nansum(post*drv_list[None,:], axis=1)
    plt.plot(vsini_list, vsisipost / np.nanmax(vsisipost))
    plt.xlabel("vsin(i) (km/s)")
    # plt.subplot(2, 3, 6)
    # plt.imshow(logpostout[0,:,:],interpolation="nearest",origin="lower",extent=[rv_list[0],rv_list[-1],vsini_list[0],vsini_list[-1]])
    # plt.xlabel("RV (km/s)")
    # plt.xlabel("vsin(i) (km/s)")
    # plt.colorbar()

    if 1:
        print("Saving " + out.replace(".fits",".png"))
        plt.savefig(out.replace(".fits",".png"))
        plt.savefig(out.replace(".fits",".pdf"))


    # plt.figure(3)
    # plt.subplot(3, 1, 1)
    # plt.imshow(fluxout[0, 0, :, :] , interpolation="nearest", origin="lower")
    # plt.subplot(3, 1, 2)
    # plt.imshow(post, interpolation="nearest", origin="lower")
    # plt.subplot(3, 1, 3)
    # plt.imshow(logpostout[0, :, :], interpolation="nearest", origin="lower")

    plt.show()
