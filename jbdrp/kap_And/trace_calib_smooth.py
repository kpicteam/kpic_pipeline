

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
from scipy.signal import medfilt
import pandas as pd

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    # x5, x4, x3, x2, x1, x0
    # -6.347647789728822e-23, -8.763050043241049e-18, -1.2137086057712525e-13, 9.833295632348853e-10, 2.205533800453195e-05, 2.438475434083729
    # -1.5156054451627633e-19, 8.603654109299864e-16, -1.80716247518748e-12, 2.238166690200087e-09, 2.1123747999061772e-05, 2.3625134057175177
    # 1.2043489185290055e-19, -6.07134563401576e-16, 1.017386463625886e-12, -9.26853752638751e-11, 2.1241653129345714e-05, 2.291160322496969
    # 3.968104754663601e-19, -1.3124938289815753e-15, 1.112571593986215e-12, 6.134905626295576e-10, 2.017316485869667e-05, 2.2242225934551327
    # 8.041271039507505e-20, -1.188553085654241e-15, 3.294776997678977e-12, -2.9632213749811383e-09, 2.1429663086447367e-05, 2.1608253921854454
    # 9.523371077287098e-20, 1.9255684438936675e-16, -1.7740532731738176e-12, 3.2497136280253542e-09, 1.785770822240155e-05, 2.10169669134709
    # -1.1921491225993055e-20, -2.4710800082806276e-17, -4.745847155094822e-13, 2.324820920356842e-09, 1.6978632482708987e-05, 2.0455151483483864
    # -1.0581425110234265e-20, 2.1172786131371726e-16, -7.916021064868458e-13, 1.4427033600147576e-09, 1.7857955925450578e-05, 1.9918890220516599
    # -7.67607806109257e-19, 3.893162857192925e-15, -6.697099419050439e-12, 5.105147905651209e-09, 1.6235110771773918e-05, 1.9414551546740444

    mykpicdir = "/scr3/jruffio/data/kpic/"
    kap_And_dir = os.path.join(mykpicdir,"kap_And_20191107")

    #background
    background60_med_filename = os.path.join(kap_And_dir,"calib","background60_med.fits")
    hdulist = pyfits.open(background60_med_filename)
    background60_med = hdulist[0].data
    background60_badpixmap_filename = os.path.join(kap_And_dir,"calib","background60_badpixmap.fits")
    hdulist = pyfits.open(background60_badpixmap_filename)
    background60_badpixmap = hdulist[0].data
    ny,nx = background60_med.shape

    # plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.imshow(background60_med*background60_badpixmap,interpolation="nearest",origin="lower")
    # plt.subplot(1,2,2)
    # plt.imshow(background60_badpixmap,interpolation="nearest",origin="lower")
    # plt.show()


    # star_filelist = glob(os.path.join(kap_And_dir, "star", "*.fits"))
    # #fibers: [1, 2, 3, 1, 2, 3, 2, 2, 2, 2, 2]
    # # planet is fiber 2
    # # star_filelist = ["/scr3/jruffio/data/kpic/kap_And_20191107/star/nspec191107_0041.fits",
    # #                  "/scr3/jruffio/data/kpic/kap_And_20191107/star/nspec191107_0042.fits",
    # #                  "/scr3/jruffio/data/kpic/kap_And_20191107/star/nspec191107_0043.fits"]
    #
    # fiber1 = [[70,150],[260,330],[460,520],[680,720],[900,930],[1120,1170],[1350,1420],[1600,1690],[1870,1980]]
    # fiber2 = [[50,133],[240,320],[440,510],[650,710],[880,910],[1100,1150],[1330,1400],[1580,1670],[1850,1960]]
    # fiber3 = [[30,120],[220,300],[420,490],[640,690],[865,890],[1090,1130],[1320,1380],[1570,1650],[1840,1940]]
    # fibers = {1:fiber1,2:fiber2,3:fiber3}
    # fiber1_template = np.zeros(2048)
    # for x1,x2 in fiber1:
    #     fiber1_template[x1+10:x2-10] = 1
    # fiber2_template = np.zeros(2048)
    # for x1,x2 in fiber2:
    #     fiber2_template[x1+10:x2-10] = 1
    # fiber3_template = np.zeros(2048)
    # for x1,x2 in fiber3:
    #     fiber3_template[x1+10:x2-10] = 1
    # fibers_template = {1: fiber1_template, 2: fiber2_template, 3: fiber3_template}
    #
    # star_list = []
    # fiber_list = []
    # for filename in star_filelist:
    #     hdulist = pyfits.open(filename)
    #     star_im = hdulist[0].data.T
    #     hdulist.close()
    #
    #     star_im_skysub = star_im-background60_med
    #     star_list.append(star_im_skysub)
    #     # plt.figure(1)
    #     # plt.imshow(star_im_skysub*background60_badpixmap,interpolation="nearest",origin="lower")
    #     # plt.clim([0,20])
    #     flattened = np.nanmean(star_im_skysub*background60_badpixmap,axis=1)
    #     fiber_list.append(np.argmax([np.nansum(fiber1_template * flattened),
    #                        np.nansum(fiber2_template*flattened),
    #                        np.nansum(fiber3_template*flattened)])+1)
    # #     plt.plot(flattened,label=os.path.basename(filename))
    # # plt.plot(fiber1_template*200,label="1")
    # # plt.plot(fiber2_template*300,label="2")
    # # plt.plot(fiber3_template*400,label="3")
    # # plt.legend()
    # # print(fiber_list)
    # # plt.show()
    # star_cube = np.array(star_list)

    ##calculate traces, FWHM, stellar spec for each fibers
    # fiber,order,x,[y,yerr,FWHM,FHWMerr,flux,fluxerr],
    for fiber_num in np.arange(1,4):
        out = os.path.join(kap_And_dir, "calib", "trace_calib_fiber{0}.fits".format(fiber_num))
        hdulist = pyfits.open(out)
        old_trace_calib = hdulist[0].data
        out = os.path.join(kap_And_dir, "calib", "residuals_fiber{0}.fits".format(fiber_num))
        hdulist = pyfits.open(out)
        old_residuals = hdulist[0].data

        # background60_badpixmap[np.where(np.abs(old_residuals)>3)]=np.nan

        polyfit_trace_calib = np.zeros(old_trace_calib.shape)
        smooth_trace_calib = np.zeros(old_trace_calib.shape)
        # paras0 = [A, w, y0, B, rn] or [A, w, y0, B, rn, g]?
        x = np.arange(0, nx)
        for order_id in range(9):
            for para_id in range(5):
                print(order_id,para_id)
                vec= old_trace_calib[order_id,:,para_id]
                vec_cp = copy(vec)

                vec_lpf = np.array(pd.DataFrame(vec_cp).rolling(window=301,center=True).median().interpolate(method="linear"))[:,0]#.fillna(method="bfill").fillna(method="ffill"))[:,0]
                wherenan_vec_lpf = np.where(np.isnan(vec_lpf))
                vec_lpf = np.array(pd.DataFrame(vec_lpf).fillna(method="bfill").fillna(method="ffill"))[:,0]
                vec_hpf = vec-vec_lpf
                vec_hpf_std =mad_std(vec_hpf[np.where(np.isfinite(vec_hpf))])
                vec[np.where(np.abs(vec_hpf)>5*vec_hpf_std)] = np.nan

                # plt.plot(vec)
                # plt.plot(np.array(pd.DataFrame(vec_cp).rolling(window=301,center=True).median()))
                # plt.plot(np.array(pd.DataFrame(vec_cp).rolling(window=301,center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")),"--")
                # plt.show()

                # vec_lpf[np.where(np.isnan(vec))] = np.nan
                wherefinitevec = np.where(np.isfinite(vec))
                polyfit_trace_calib[order_id,:,para_id] = np.polyval(np.polyfit(x[wherefinitevec],vec[wherefinitevec],5),x)

                # vec_lpf[wherenan_vec_lpf] = polyfit_trace_calib[order_id,wherenan_vec_lpf[0],para_id]
                smooth_trace_calib[order_id,:,para_id] = vec_lpf

                # # [A, w, y0, B, rn]
                if para_id == 1:
                    plt.plot(vec)
                    plt.plot(vec_lpf)
                    # plt.plot(vec_hpf)
                    # plt.plot(np.ones(vec_hpf.shape)*vec_hpf_std)
                    plt.plot(np.polyval(np.polyfit(x[np.where(np.isfinite(vec))],vec[np.where(np.isfinite(vec))],5),x))
                    plt.show()

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=polyfit_trace_calib))
        out = os.path.join(kap_And_dir, "calib", "trace_calib_polyfit_fiber{0}.fits".format(fiber_num))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=smooth_trace_calib))
        out = os.path.join(kap_And_dir, "calib", "trace_calib_smooth_fiber{0}.fits".format(fiber_num))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
    exit()