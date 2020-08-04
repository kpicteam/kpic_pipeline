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

    return LPF_myvec,HPF_myvec



if __name__ == "__main__":
    data_dir = "/scr3/jruffio/data/kpic/HDC_workshop_data/"
    BD = "B" #"A" or "B" for 2M0746A or 2M0746B
    for BD in ["A","B"]:
        # selec_orders = [0,1,2,3,4,5,6,7,8]
        selec_orders = [6]
        cutoff = 20
        c_kms = 299792.458
        rv_list = np.concatenate([np.arange(-300,-40,3),np.arange(-40,40,0.5),np.arange(40,300,3)], axis=0)
        rv_list = np.arange(-200,60,1)[:,None]


        # read the companion signal spectra and combine
        if 1:
            hdulist = pyfits.open(os.path.join(data_dir, "2M0746"+BD+"_spectra.fits"))
            science_spectra = hdulist[0].data[:,selec_orders,50:2000]
            hdulist = pyfits.open(os.path.join(data_dir, "2M0746"+BD+"_spectra_err.fits"))
            science_err = hdulist[0].data[:,selec_orders,50:2000]

            hdulist = pyfits.open(os.path.join(data_dir, "wvs.fits"))
            science_wvs = hdulist[0].data[selec_orders,50:2000]
            N_order = science_wvs.shape[0]

            combined_science_spectra = np.nansum(science_spectra,axis=0)
            combined_science_err = 1/np.sqrt(np.nansum(1/science_err**2,axis=0))

            where_nans = np.where(np.nansum(np.isnan(science_spectra) != 0,axis=0))

            combined_science_spectra[where_nans] = np.nan
            combined_science_err[where_nans] = np.nan

        # read the A0 star spectra and combine
        if 1:
            hdulist = pyfits.open(os.path.join(data_dir, "A0_star_spectra.fits"))
            star_spectra = hdulist[0].data
            star_spectra = star_spectra[:, selec_orders, 50:2000]
            hdulist = pyfits.open(os.path.join(data_dir, "A0_star_spectra_err.fits"))
            star_err = hdulist[0].data
            star_err = star_err[:, selec_orders, 50:2000]


            # combine A0 stellar spectra
            med_star_spec = np.nanmedian(star_spectra, axis=0)
            for k in range(star_spectra.shape[0]):
                for l in range(N_order):
                    wherefinite = np.where(np.isfinite(star_spectra[k, l, :]) * np.isfinite(med_star_spec[l, :]))[0]
                    scaling = np.nansum(star_spectra[k, l, wherefinite] * med_star_spec[l, wherefinite]) / np.nansum(
                        star_spectra[k, l, wherefinite] ** 2)
                    star_spectra[k, l, :] = star_spectra[k, l, :] * scaling
                    star_err[k, l, :] = star_err[k, l, :] * scaling
            # combined_star_spectra = np.nansum(star_spectra/star_err**2,axis=0)/np.nansum(1/star_err**2,axis=0) # This is not working, there is a problem somewhere
            combined_star_spectra = np.nansum(star_spectra, axis=0)
            combined_star_err = 1 / np.sqrt(np.nansum(1 / star_err ** 2, axis=0))

            where_nans = np.where(np.nansum(np.isnan(star_spectra) != 0, axis=0))

            combined_star_spectra[where_nans] = np.nan
            combined_star_err[where_nans] = np.nan

        where_nans = np.where((np.isfinite(combined_star_spectra) == 0)+\
                              (np.isfinite(combined_science_spectra) == 0)+ \
                              (np.isfinite(combined_star_err) == 0) + \
                              (np.isfinite(combined_science_err) == 0))

        combined_star_spectra[where_nans] = np.nan
        combined_star_err[where_nans] = np.nan
        combined_science_spectra[where_nans] = np.nan
        combined_science_err[where_nans] = np.nan

        # Read the linewidth and pixelwidth in units of wavelength
        if 1:
            hdulist = pyfits.open(os.path.join(data_dir, "trace_calib_polyfit_fiber2.fits"))
            science_lw = np.abs(hdulist[0].data[:, :, 1])
            science_lw = science_lw[selec_orders,50:2000]
            tmp_dwvs = science_wvs[:, 1:science_wvs.shape[-1]] - science_wvs[:, 0:science_wvs.shape[-1]-1]
            tmp_dwvs = np.concatenate([tmp_dwvs, tmp_dwvs[:, -1][:, None]], axis=1)
            science_lw_wvunit = science_lw * tmp_dwvs
            line_width_func = interp1d(np.ravel(science_wvs), np.ravel(science_lw_wvunit), bounds_error=False,
                                       fill_value=np.nan)
            pixel_width_func = interp1d(np.ravel(science_wvs), np.ravel(tmp_dwvs), bounds_error=False, fill_value=np.nan)

        specpool = mp.Pool(processes=30)
        if 1: # read BD template
            with open(os.path.join(data_dir, "lte018-5.0-0.0a+0.0.BT-Settl.spec.7"), 'r') as f:
                model_wvs = []
                model_fluxes = []
                for line in f.readlines():
                    line_args = line.strip().split()
                    model_wvs.append(float(line_args[0]))
                    model_fluxes.append(float(line_args[1].replace('D', 'E')))
            model_wvs = np.array(model_wvs)/1.e4
            model_fluxes = np.array(model_fluxes)
            model_fluxes = 10 ** (model_fluxes - 8)

            # crop model around relevant wavelength
            crop_plmodel = np.where((model_wvs>1.8-(2.6-1.8)/2)*(model_wvs<2.6+(2.6-1.8)/2))
            model_wvs = model_wvs[crop_plmodel]
            model_fluxes = model_fluxes[crop_plmodel]

            # Convolve spectrum using linewidth
            pl_line_widths = np.array(pd.DataFrame(line_width_func(model_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
            pl_pixel_widths = np.array(pd.DataFrame(pixel_width_func(model_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
            planet_convspec = convolve_spectrum_line_width(model_wvs,model_fluxes,pl_line_widths,mypool=specpool)
            planet_convspec = convolve_spectrum_pixel_width(model_wvs,planet_convspec,pl_pixel_widths,mypool=specpool)
            planet_convspec /= np.nanmean(planet_convspec)

            # generate a spline interpolator of the template
            from scipy import interpolate
            BD_template_spline = interpolate.splrep(model_wvs, planet_convspec)

        # Read model of the A0 star
        if 1:
            phoenix_wv_filename = os.path.join(data_dir,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
            with pyfits.open(phoenix_wv_filename) as hdulist:
                phoenix_wvs = hdulist[0].data/1.e4
            crop_phoenix = np.where((phoenix_wvs>1.8-(2.6-1.8)/2)*(phoenix_wvs<2.6+(2.6-1.8)/2))
            phoenix_wvs = phoenix_wvs[crop_phoenix]
            phoenix_model_host_filename = glob(os.path.join(data_dir, "kap_And" + "*.fits"))[0]
            with pyfits.open(phoenix_model_host_filename) as hdulist:
                phoenix_A0 = hdulist[0].data[crop_phoenix]

            phoenix_line_widths = np.array(pd.DataFrame(line_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
            phoenix_pixel_widths = np.array(pd.DataFrame(pixel_width_func(phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
            phoenix_A0_conv = convolve_spectrum_line_width(phoenix_wvs,phoenix_A0,phoenix_line_widths,mypool=specpool)
            phoenix_A0_conv = convolve_spectrum_pixel_width(phoenix_wvs,phoenix_A0_conv,phoenix_pixel_widths,mypool=specpool)
            phoenix_A0_conv /= np.nanmean(phoenix_A0_conv)

            from scipy.interpolate import interp1d
            phoenix_A0_func = interp1d(phoenix_wvs, phoenix_A0_conv, bounds_error=False, fill_value=np.nan)


        # calculate transmission
        if 1:
            where_data_nans = np.where(np.isnan(combined_science_spectra))
            transmission = combined_star_spectra/phoenix_A0_func(science_wvs)
            transmission[where_data_nans] = np.nan

        if 0: #injecting fake planet
            c_kms = 299792.458
            rv0 = 10
            print(combined_science_spectra.shape)
            print(np.nanmedian(combined_science_spectra))
            ampl = 0.5*np.nanmedian(combined_science_spectra)
            combined_science_spectra_ori = copy(combined_science_spectra)
            fake = interpolate.splev(science_wvs * (1 - rv0 / c_kms), kap_And_spec_spline, der=0) * transmission
            fake[where_data_nans] = np.nan
            fake = ampl * fake/np.nanmedian(fake)
            combined_science_spectra += fake


        # high pass filter the science spectra
        if 1:
            combined_science_spectra_hpf = np.zeros(combined_science_spectra.shape)
            combined_science_spectra_lpf = np.zeros(combined_science_spectra.shape)
            for order_id in range(N_order):
                p = combined_science_spectra[order_id,:]
                p_lpf, p_hpf = LPFvsHPF(p, cutoff=cutoff)
                combined_science_spectra_lpf[order_id,:] = p_lpf
                combined_science_spectra_hpf[order_id,:] = p_hpf
                # plt.figure(1,figsize=(8,4))
                # plt.plot(science_wvs[order_id,:], p,label="star*transmission")
                # plt.plot(science_wvs[order_id,:], p_lpf,label="Low-pass",linewidth=5)
                # plt.plot(science_wvs[order_id,:], p_hpf,label="High-pass")
                # plt.xlabel(r"$\lambda$ $\mu$m")
                # plt.legend()
                # plt.savefig(os.path.join(data_dir,"out","HPF_data.png"),bbox_inches='tight')
                # plt.show()


        # model of the host star
        if 1:
            m2 = phoenix_A0_func(science_wvs)*transmission
            # m2 = transmission
            for order_id in range(N_order):
                m2_tmp_lpf,m2_tmp_hpf= LPFvsHPF(m2[order_id, :], cutoff=cutoff)
                # plt.figure(1,figsize=(8,4))
                # plt.plot(science_wvs[order_id,:], m2[order_id, :],label="star*transmission")
                # plt.plot(science_wvs[order_id,:], m2_tmp_lpf,label="Low-pass",linewidth=5)
                # plt.plot(science_wvs[order_id,:], m2_tmp_hpf,label="High-pass")
                # plt.xlabel(r"$\lambda$ $\mu$m")
                # plt.legend()
                # plt.savefig(os.path.join(data_dir,"out","HPF_star.png"),bbox_inches='tight')
                # plt.show()
                m2[order_id,:] =  m2_tmp_hpf/m2_tmp_lpf* combined_science_spectra_lpf[order_id,:]


        # forward model the data and fit as a function of planet RV
        out_dAIC = np.zeros(rv_list.shape)
        out_ampl = np.zeros(rv_list.shape)
        out_logpost = np.zeros(rv_list.shape)
        for rv_id, rv in enumerate(rv_list):
            print(rv_id,np.size(rv_list))

            m1 = interpolate.splev(science_wvs*(1-rv/c_kms),BD_template_spline,der=0)*transmission
            for order_id in range(N_order):
                m1[order_id,:] =  LPFvsHPF(m1[order_id,:], cutoff=cutoff)[1]
            # combined_science_spectra_hpf[np.where(np.isnan(m1)*np.isnan(m2))] = np.nan

            ravelHPFdata = np.ravel(combined_science_spectra_hpf)
            ravelwvs = np.ravel(science_wvs)

            where_data_finite = np.where(np.isfinite(ravelHPFdata))
            ravelHPFdata = ravelHPFdata[where_data_finite]
            ravelwvs = ravelwvs[where_data_finite]
            sigmas_vec = np.ravel(combined_science_err)[where_data_finite]
            logdet_Sigma = np.sum(2*np.log(sigmas_vec))

            m1_ravel = np.ravel(m1)[where_data_finite]
            m2_ravel = np.ravel(m2)[where_data_finite]
            HPFmodel_H0 = m2_ravel[:,None]
            HPFmodel = np.concatenate([m1_ravel[:,None], m2_ravel[:,None]], axis=1)

            HPFparas, HPFchi2, rank, s = np.linalg.lstsq(HPFmodel/sigmas_vec[:,None], ravelHPFdata/sigmas_vec, rcond=None)
            HPFparas_H0, HPFchi2_H0, rank, s = np.linalg.lstsq(HPFmodel_H0/sigmas_vec[:,None], ravelHPFdata/sigmas_vec, rcond=None)
            print("H1",HPFparas)
            print("H0",HPFparas_H0)
            plt.figure(1, figsize=(8, 4))
            # plt.fill_between(ravelwvs,ravelHPFdata-sigmas_vec,ravelHPFdata+sigmas_vec, label = "error",alpha=0.2)
            plt.plot(ravelwvs,ravelHPFdata,label = "data",alpha= 1)
            plt.plot(ravelwvs,HPFparas[0]*m1_ravel, label = "planet",alpha= 1)
            plt.plot(ravelwvs,HPFparas[1]*m2_ravel, label = "star",alpha= 1)
            # plt.plot(ravelwvs,ravelHPFdata-HPFparas[1]*m2_ravel, label = "res",alpha= 0.5)
            plt.xlabel(r"$\lambda$ $\mu$m")
            plt.legend()
            plt.savefig(os.path.join(data_dir,"out","data_badfit.png"),bbox_inches='tight')
            plt.show()
            exit()

            data_model = np.dot(HPFmodel, HPFparas)
            data_model_H0 = np.dot(HPFmodel_H0, HPFparas_H0)
            deltachi2 = 0  # chi2ref-np.sum(ravelHPFdata**2)
            ravelresiduals = ravelHPFdata - data_model
            ravelresiduals_H0 = ravelHPFdata - data_model_H0
            HPFchi2 = np.nansum((ravelresiduals) ** 2)
            HPFchi2_H0 = np.nansum((ravelresiduals_H0) ** 2)

            Npixs_HPFdata = HPFmodel.shape[0]
            minus2logL_HPF = Npixs_HPFdata * (1 + np.log(HPFchi2 / Npixs_HPFdata) + logdet_Sigma + np.log(2 * np.pi))
            minus2logL_HPF_H0 = Npixs_HPFdata * (1 + np.log(HPFchi2_H0 / Npixs_HPFdata) + logdet_Sigma + np.log(2 * np.pi))
            AIC_HPF = 2 * (HPFmodel.shape[-1]) + minus2logL_HPF
            AIC_HPF_H0 = 2 * (HPFmodel_H0.shape[-1]) + minus2logL_HPF_H0

            out_dAIC[rv_id] = AIC_HPF_H0-AIC_HPF
            out_ampl[rv_id] = HPFparas[0]
            slogdet_icovphi0 = np.linalg.slogdet(np.dot(HPFmodel.T,HPFmodel))
            out_logpost[rv_id] =  -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0[1] - (Npixs_HPFdata - HPFmodel.shape[-1] + 2 - 1) / (2) * np.log(HPFchi2)


        # plt.subplot(1,3,1)
        # plt.plot(rv_list,out_dAIC)
        # plt.xlabel("RV (km/s)")
        plt.figure(1,figsize=(4,4))
        plt.plot(rv_list,out_ampl,label=BD)
        plt.figure(2,figsize=(4,4))
        plt.plot(rv_list,np.exp(out_logpost-np.max(out_logpost)),label=BD)

    plt.figure(1)
    plt.legend()
    plt.xlabel("RV (km/s)")
    plt.ylabel("Planet Amplitude ($\sim$ CCF)")
    plt.savefig(os.path.join(data_dir, "out", "CCF_allorders.png"), bbox_inches='tight')
    plt.figure(2)
    plt.legend()
    plt.xlabel("RV (km/s)")
    plt.ylabel("Posterior")
    plt.xlim([10,30])
    plt.savefig(os.path.join(data_dir, "out", "posterior_allorders.png"), bbox_inches='tight')
    plt.show()
