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

    mykpicdir = "/scr3/jruffio/data/kpic/"
    kap_And_dir = os.path.join(mykpicdir, "kap_And_20191107")


    for fiber_num in np.arange(1,4):
        star_filelist = glob(os.path.join(kap_And_dir, "calib", "*_flux_extract_f{0}.fits".format(fiber_num)))
        star_filelist.sort()
        # print(star_filelist)
        # exit()

        star_spectra = np.zeros((len(star_filelist),9,2048))
        star_wvs = np.zeros((len(star_filelist),9,2048))
        star_background = np.zeros((len(star_filelist),9,2048))
        star_rn = np.zeros((len(star_filelist),9,2048))
        for fileid,filename in enumerate(star_filelist):
            out = os.path.join(kap_And_dir, "calib", os.path.basename(filename))
            out = glob(out)[0]
            hdulist = pyfits.open(out)
            extract_paras = hdulist[0].data
            # print(extract_paras.shape)
            # exit()
            out = os.path.join(kap_And_dir, "calib", os.path.basename(filename).replace("_flux_extract_f","_flux_extract_res_f"))
            out = glob(out)[0]
            hdulist = pyfits.open(out)
            residuals = hdulist[0].data

            fiber_num = int(out.replace(".fits","")[-1])


            for order_id in range(9):

                # vec = extract_paras[order_id,:,1]
                # vec_lpf = np.array(pd.DataFrame(vec).rolling(window=301, center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
                # vec_hpf = vec - vec_lpf
                # vec_hpf_std = mad_std(vec_hpf[np.where(np.isfinite(vec_hpf))])
                # where_nans = np.where(np.abs(vec_hpf) > 5 * vec_hpf_std)
                #
                # # vec[where_nans] = np.nan
                # # vec_lpf[np.where(np.isnan(vec))] = np.nan
                # # wherefinitevec = np.where(np.isfinite(vec))
                # # polyfit_trace_calib[order_id, :, para_id] = np.polyval(
                # #     np.polyfit(x[wherefinitevec], vec[wherefinitevec], 2), x)

                a = extract_paras[order_id,:,0]
                # a[where_nans] = np.nan
                star_spectra[fileid,order_id,:] = a
                a = extract_paras[order_id,:,1]
                # a[where_nans] = np.nan
                star_background[fileid,order_id,:] = a
                a = extract_paras[order_id,:,2]
                # a[where_nans] = np.nan
                star_rn[fileid,order_id,:] = np.abs(a)

        star_spectra[:,:,0:50] = np.nan
        star_spectra[:,:,(2048-50)::] = np.nan
        star_background[:,:,0:50] = np.nan
        star_background[:,:,(2048-50)::] = np.nan
        star_rn[:,:,0:50] = np.nan
        star_rn[:,:,(2048-50)::] = np.nan


        plt.figure(1)
        # star_spectra = star_spectra/np.nanmean(star_spectra,axis=2)[:,:,None]
        cp_star_spectra = copy(star_spectra)
        print(np.nanmean(star_spectra,axis=2))
        star_med_spec = np.nanmedian(star_spectra,axis=0)
        for k in range(len(star_filelist)):
            for l in range(9):
                wherefinite = np.where(np.isfinite(star_spectra[k, l, :])*np.isfinite(star_med_spec[l, :]))[0]
                star_spectra[k, l, :] = star_spectra[k, l, :]*np.nansum(star_spectra[k, l, wherefinite]*star_med_spec[ l, wherefinite])/np.nansum(star_spectra[k, l, wherefinite]**2)
        star_med_spec = np.nanmedian(star_spectra,axis=0)

        # print(np.nanmean(star_med_spec,axis=1))
        # exit()
        # star_rn_med = np.tile(np.nanmedian(star_rn, axis=2)[:,:,None],(1,1,2048))
        #
        # star_spectra[np.where(star_rn>2*star_rn_med)]=np.nan
        # cp_star_spectra[np.where(star_rn>2*star_rn_med)]=np.nan
        # star_rn[np.where(star_rn>2*star_rn_med)]=np.nan

        if len(star_filelist) >= 2:
            cp_star_rn = copy(star_rn)
            star_rn = star_rn/np.nanmedian(star_rn,axis=2)[:,:,None]
            star_rn_med = np.nanmedian(star_rn,axis=0)
            star_rn_res = (star_rn - star_rn_med[None, :, :])
            star_rn_std = np.zeros(star_rn.shape)
            for k in range(len(star_filelist)):
                for l in range(9):
                    tmp = star_rn_res[k,l,:]
                    star_rn_std[k,l,:] = mad_std(tmp[np.where(np.isfinite(tmp))])
            # star_rn_std = np.tile(np.nanstd(star_rn_res, axis=2)[:, :, None], (1, 1, 2048))
            star_rn_ratio = np.tile(np.nanmax(np.abs(star_rn_res) / star_rn_std, axis=0)[None, :, :],(len(star_filelist), 1, 1))
            # star_rn_ratio = np.abs(star_rn_res) / star_rn_std
            star_rn[np.where(star_rn_ratio > 5)] = np.nan
            cp_star_rn[np.where(star_rn_ratio > 5)] = np.nan
            star_spectra[np.where(star_rn_ratio > 5)] = np.nan
            cp_star_spectra[np.where(star_rn_ratio > 5)] = np.nan

        # star_rn_med = np.nanmedian(star_rn, axis=2)
        # star_rn_res = star_rn-star_rn_med[:,:,None]
        # star_rn_std = np.tile(np.nanstd(star_rn_res,axis=2)[:,:,None],(1,1,2048))
        # star_rn_ratio = np.tile(np.nanmax(np.abs(star_rn_res)/star_rn_std,axis=0)[None,:,:],(len(star_filelist),1,1))

        # star_spectra[np.where(star_rn_ratio>3)]=np.nan

        # star_background_med = np.nanmedian(star_background, axis=2)
        # star_background_res = star_background-star_background_med[:,:,None]
        # star_background_std = np.tile(np.nanstd(star_background_res,axis=2)[:,:,None],(1,1,2048))
        # star_background_ratio = np.tile(np.nanmax(np.abs(star_background_res)/star_background_std,axis=0)[None,:,:],(len(star_filelist),1,1))
        #
        # star_spectra[np.where(star_background_ratio>3)]=np.nan

        star_med_spec = np.nanmedian(star_spectra,axis=0)


        if len(star_filelist)>=2:
            # star_spectra_ratio = np.abs((star_spectra-star_med_spec[None,:,:])/star_med_spec[None,:,:])
            # # star_spectra_std = np.tile(np.nanstd(star_spectra_res,axis=2)[:,:,None],(1,1,2048))
            # # star_spectra_ratio = np.tile(np.nanmax(np.abs(star_spectra_res)/star_spectra_std,axis=0)[None,:,:],(len(star_filelist),1,1))
            # star_spectra[np.where(star_spectra_ratio>0.05)]=np.nan
            # cp_star_spectra[np.where(star_spectra_ratio>0.05)]=np.nan

            star_spectra_res = (star_spectra-star_med_spec[None,:,:])
            star_spectra_std= np.zeros(star_spectra.shape)
            for k in range(len(star_filelist)):
                for l in range(9):
                    tmp = star_spectra_res[k,l,:]
                    star_spectra_std[k,l,:] = mad_std(tmp[np.where(np.isfinite(tmp))])
            # spectra_std = np.tile(np.nanstd(spectra_res,axis=2)[:,:,None],(1,1,2048))
            spectra_ratio = np.tile(np.nanmax(np.abs(star_spectra_res)/star_spectra_std,axis=0)[None,:,:],(len(star_filelist),1,1))
            # spectra_ratio = np.abs(spectra_res)/spectra_std
            star_spectra[np.where(spectra_ratio>5)]=np.nan
            cp_star_spectra[np.where(spectra_ratio>5)]=np.nan
        star_med_spec = np.nanmedian(star_spectra,axis=0)

        wvs_per_orders = np.zeros((9,2048))
        for order_id in range(9):
            wvs = np.polyval(polycoefs_wavsol_jason[8 - order_id, :], np.arange(0, 2048))
            wvs_per_orders[order_id,:] = wvs

        # exit()
        # for fileid, filename in enumerate(star_filelist):
        #     for order_id in range(9):
        #         plt.subplot(9, 1, 9-order_id )
        #         plt.plot(wvs_per_orders[order_id,:],cp_star_spectra[fileid,order_id,:],alpha=0.5,linewidth=1)
        for order_id in range(9):
            plt.subplot(9, 1, 9-order_id)
            plt.plot(wvs_per_orders[order_id,:],star_med_spec[order_id,:],linestyle="--",linewidth=2)
        # plt.tight_layout()



        # plt.figure(2)
        # for fileid, filename in enumerate(star_filelist):
        #     for order_id in range(9):
        #         plt.subplot(9, 1, 9-order_id)
        #         plt.plot(cp_star_rn[fileid,order_id,:],alpha=0.5,linewidth=1)
        #         # plt.ylim([0,5])
        # plt.figure(3)
        # for fileid, filename in enumerate(star_filelist):
        #     for order_id in range(9):
        #         plt.subplot(9, 1, 9-order_id)
        #         plt.plot(star_background_ratio[fileid,order_id,:],alpha=0.5,linewidth=1)
        #
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=cp_star_spectra))
        out = os.path.join(kap_And_dir, "calib", os.path.basename(star_filelist[0]).replace( "_flux_extract","_star_spectra"))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=cp_star_rn))
        out = os.path.join(kap_And_dir, "calib", os.path.basename(star_filelist[0]).replace( "_flux_extract","_star_spectraerr"))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=wvs_per_orders))
        out = os.path.join(kap_And_dir, "calib", os.path.basename(star_filelist[0]).replace( "_flux_extract","_star_wvs"))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()
    plt.show()
    exit()