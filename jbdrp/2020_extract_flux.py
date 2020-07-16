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
from utils_2020.extract_flux import extract_flux
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u

from astropy.utils import iers
from astropy.utils.iers import conf as iers_conf
print(iers_conf.iers_auto_url)
#default_iers = iers_conf.iers_auto_url
#print(default_iers)
iers_conf.iers_auto_url = 'https://datacenter.iers.org/data/9/finals2000A.all'
iers_conf.iers_auto_url_mirror = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'
iers.IERS_Auto.open()  # Note the URL

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    mykpicdir = "/scr3/jruffio/data/kpic/"
    fitbackground = True
    bad_pixel_fraction = 0.03

    no_background = False
    # mydir = os.path.join(mykpicdir,"20200607_5_Vul") #
    # mydir = os.path.join(mykpicdir,"20200607_HIP_95771") #
    # mydir = os.path.join(mykpicdir,"20200607_ups_Her") #
    # mydir = os.path.join(mykpicdir,"20200607_HIP_81497") #
    # mydir = os.path.join(mykpicdir,"20200608_d_Sco")#
    # mydir = os.path.join(mykpicdir,"20200608_zet_Aql") #
    # mydir = os.path.join(mykpicdir,"20200608_kap_And") #
    # mydir = os.path.join(mykpicdir,"20200608_kap_And_B") #
    # mydir = os.path.join(mykpicdir,"20200608_HR_7672_B") #
    # mydir = os.path.join(mykpicdir,"20200608_HIP_81497_30s")#
    # mydir = os.path.join(mykpicdir,"20200608_HIP_81497_7.5s")#
    # mydir = os.path.join(mykpicdir,"20200609_ups_Her")#
    # mydir = os.path.join(mykpicdir,"20200609_d_Sco")#
    # mydir = os.path.join(mykpicdir,"20200609_kap_And") #
    # mydir = os.path.join(mykpicdir,"20200609_HIP_81497") #
    # mydir = os.path.join(mykpicdir,"20200609_kap_And_B") #
    # mydir = os.path.join(mykpicdir,"20200609_HR_7672")#
    # mydir = os.path.join(mykpicdir,"20200609_HR_7672_B")#
    # mydir = os.path.join(mykpicdir,"20200609_zet_Aql")#
    # mydir = os.path.join(mykpicdir,"20200609_ROXs_12B")
    # mydir = os.path.join(mykpicdir,"20200701_ups_Her")
    # mydir = os.path.join(mykpicdir,"20200701_HIP_81497")
    # mydir = os.path.join(mykpicdir,"20200701_ROXs_42Bb")
    # mydir = os.path.join(mykpicdir,"20200701_15_Sgr")
    # mydir = os.path.join(mykpicdir,"20200701_HR_8799")
    # mydir = os.path.join(mykpicdir,"20200701_HR_8799_c")
    # mydir = os.path.join(mykpicdir,"20200702_ups_Her")
    # mydir = os.path.join(mykpicdir,"20200702_HIP_81497")
    # mydir = os.path.join(mykpicdir,"20200702_d_Sco")
    # mydir = os.path.join(mykpicdir,"20200702_15_Sgr")
    # mydir = os.path.join(mykpicdir,"20200702_ROXs_42Bb")
    # mydir = os.path.join(mykpicdir,"20200702_ROXs_42B")
    # mydir = os.path.join(mykpicdir,"20200702_ROXs_42B_daytime")
    # mydir = os.path.join(mykpicdir,"20200702_ROXs_42B_fit")
    # mydir = os.path.join(mykpicdir,"20200702_ROXs_42B_chopping")
    # mydir = os.path.join(mykpicdir,"20200702_HR_8799")
    # mydir = os.path.join(mykpicdir,"20200702_HR_8799_d")
    # mydir = os.path.join(mykpicdir,"20200703_ups_Her")
    # mydir = os.path.join(mykpicdir,"20200703_HIP_81497")
    # mydir = os.path.join(mykpicdir,"20200703_HD_124683")
    # mydir = os.path.join(mykpicdir,"20200703_d_Sco")
    # mydir = os.path.join(mykpicdir,"20200703_ROXs_12B")
    # mydir = os.path.join(mykpicdir,"20200703_HR_8799")
    # mydir = os.path.join(mykpicdir,"20200703_HR_8799_e")
    # mydir = os.path.join(mykpicdir,"20200703_kap_And")
    mydir = os.path.join(mykpicdir,"20200703_kap_And_B")
    # mydir,no_background,fitbackground = os.path.join(mykpicdir,"20200703_ROXs_12B_chopping"),True,True

    # filelist = glob(os.path.join(mydir, "raw", "*0041*.fits"))
    filelist = glob(os.path.join(mydir, "raw", "*.fits"))
    filelist.sort()
    filelist = filelist[np.size(filelist)-3::]
    # filelist = filelist[8::]
    # print(filelist)
    # exit()
    # filelist = filelist[5::]

    addbaryrv = True

    mydate = os.path.basename(mydir).split("_")[0]


    #background
    background_med_filename = glob(os.path.join(mydir, "calib", "*background*.fits"))[0]
    hdulist = pyfits.open(background_med_filename)
    background = hdulist[0].data
    background_header = hdulist[0].header
    tint = float(background_header["TRUITIME"])
    coadds = int(background_header["COADDS"])


    persisbadpixmap_filename = glob(os.path.join(mydir,"calib","*persistent_badpix*.fits"))[0]
    hdulist = pyfits.open(persisbadpixmap_filename)
    persisbadpixmap = hdulist[0].data
    persisbadpixmap_header = hdulist[0].header
    ny,nx = persisbadpixmap.shape

    if 0: # chopping
        filelist = glob(os.path.join(mydir, "raw", "*.fits"))
        filelist.sort()
        filelistsub = [filelist[1],filelist[0],filelist[3],filelist[2]]
        for filename,filenamesub in zip(filelist,filelistsub):

            hdulist = pyfits.open(filename)
            im = hdulist[0].data
            header = hdulist[0].header
            hdulist.close()
            hdulist = pyfits.open(filenamesub)
            im_sub = hdulist[0].data
            hdulist.close()

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=im-im_sub, header=header))
            out = filename.replace(".fits","_pairsub.fits")
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()

        exit()

    if 0: # generate null backgrounds.
        hdulist = pyfits.HDUList()
        background_header["TRUITIME"] = 0
        hdulist.append(pyfits.PrimaryHDU(data=background*0, header=background_header))
        out = os.path.join(mydir,"calib", "null_background_med_tint{0}.fits".format(0))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        hdulist = pyfits.HDUList()
        persisbadpixmap_header["TRUITIME"] = 0
        hdulist.append(pyfits.PrimaryHDU(data=persisbadpixmap, header=persisbadpixmap_header))
        out = os.path.join(mydir,"calib", "null_persistent_badpix_tint{0}.fits".format(0))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()
        exit()



    trace_loc_filename = glob(os.path.join(mydir,"calib","*_trace_loc_smooth.fits"))[0]
    hdulist = pyfits.open(trace_loc_filename)
    trace_loc = hdulist[0].data
    trace_loc[np.where(trace_loc==0)] = np.nan
    print(trace_loc.shape)
    # plt.figure(1)
    # for order_id in range(9):
    #     plt.subplot(9, 1, 9-order_id)
    #     plt.plot(trace_loc[1,order_id,:],linestyle="-",linewidth=2)
    #     plt.legend()
    # plt.show()

    tracemask = np.ones(background.shape)

    trace_loc_slit = np.zeros((trace_loc.shape[0],trace_loc.shape[1],trace_loc.shape[2]))
    trace_loc_dark = np.zeros((trace_loc.shape[0]*2,trace_loc.shape[1],trace_loc.shape[2]))
    for order_id in range(9):
        for _fib in range(trace_loc.shape[0]):
            tracemask[trace_loc[_fib, order_id, :].astype(np.int),np.arange(background.shape[1])] = np.nan
            tracemask[trace_loc[_fib, order_id, :].astype(np.int)+1,np.arange(background.shape[1])] = np.nan
            tracemask[trace_loc[_fib, order_id, :].astype(np.int)-1,np.arange(background.shape[1])] = np.nan

        dy1 = np.nanmean(trace_loc[0, order_id, :] - trace_loc[1, order_id, :]) / 2
        dy2 = np.nanmean(trace_loc[0, order_id, :] - trace_loc[trace_loc.shape[0]-1, order_id, :])
        # exit()
        if np.isnan(dy1):
            dy1 = 10
        if np.isnan(dy2):
            dy2 = 40
        print(dy1, dy2)

        trace_loc_slit[0, order_id, :] = trace_loc[0, order_id, :] - dy1
        trace_loc_slit[1, order_id, :] = trace_loc[1, order_id, :] - dy1
        trace_loc_slit[2, order_id, :] = trace_loc[2, order_id, :] - dy1
        trace_loc_slit[3, order_id, :] = trace_loc[3, order_id, :] - dy1

        if order_id == 0:
            trace_loc_dark[0, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 0 * dy1
            trace_loc_dark[1, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 1 * dy1
            trace_loc_dark[2, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 2 * dy1
            trace_loc_dark[3, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 3 * dy1

            trace_loc_dark[4, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 4 * dy1
            trace_loc_dark[5, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 5 * dy1
            trace_loc_dark[6, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 6 * dy1
            trace_loc_dark[7, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 7 * dy1
        else:
            trace_loc_dark[0, order_id, :] = trace_loc[0, order_id, :] - 3 * dy2 + 3 * dy1
            trace_loc_dark[1, order_id, :] = trace_loc[1, order_id, :] - 3 * dy2 + 4 * dy1
            trace_loc_dark[2, order_id, :] = trace_loc[2, order_id, :] - 3 * dy2 + 5 * dy1
            trace_loc_dark[3, order_id, :] = trace_loc[3, order_id, :] - 3 * dy2 + 6 * dy1

            trace_loc_dark[4, order_id, :] = trace_loc[0, order_id, :] - 2 * dy2 + 2 * dy1
            trace_loc_dark[5, order_id, :] = trace_loc[1, order_id, :] - 2 * dy2 + 3 * dy1
            trace_loc_dark[6, order_id, :] = trace_loc[2, order_id, :] - 2 * dy2 + 4 * dy1
            trace_loc_dark[7, order_id, :] = trace_loc[3, order_id, :] - 2 * dy2 + 5 * dy1

    line_width_filename = glob(os.path.join(mydir,"calib","*_line_width_smooth.fits"))[0]
    hdulist = pyfits.open(line_width_filename)
    line_width = hdulist[0].data
    line_width[np.where(line_width==0)] = np.nan
    line_width_slit = line_width
    line_width_dark = np.concatenate([line_width,line_width],axis=0)
    print(line_width.shape)

    # plt.imshow(tracemask)
    # plt.show()


    for filename in filelist:
        out = os.path.join(mydir, os.path.basename(filename).replace(".fits","_fluxes.fits"))

        hdulist = pyfits.open(filename)
        im = hdulist[0].data.T[:,::-1]
        header = hdulist[0].header
        if tint != 0 and \
                (tint != float(header["TRUITIME"])>1 or coadds != int(header["COADDS"])):
            raise Exception("bad tint {0} or coadds {1}, should be {2} and {3}: ".format(float(header["TRUITIME"]),int(header["COADDS"]),tint,coadds) + filename)
        hdulist.close()

        if addbaryrv:
            keck = EarthLocation.from_geodetic(lat=19.8283*u.deg, lon=-155.4783*u.deg, height=4160*u.m)
            sc = SkyCoord(float(header["CRVAL1"]) * u.deg, float(header["CRVAL2"]) * u.deg)
            barycorr = sc.radial_velocity_correction(obstime=Time(float(header["MJD"]), format="mjd", scale="utc"), location=keck)
            header["BARYRV"] = barycorr.to(u.km/u.s).value

        # cp_im = im*tracemask*persisbadpixmap
        # where_finite = np.where(np.isfinite(cp_im))
        # print(np.nansum(cp_im[where_finite]*background[where_finite])/np.nansum(background[where_finite]**2))
        # exit()
        if not no_background:
            im_skysub = im-background#*np.nansum(cp_im[where_finite]*background[where_finite])/np.nansum(background[where_finite]**2)
        else:
            im_skysub = copy(im)
        badpixmap = persisbadpixmap#*get_badpixmap_from_laplacian(im_skysub,bad_pixel_fraction=1e-2)

        # plt.figure(1)
        # plt.imshow(im,origin="lower")
        # plt.clim([50,200])
        # for order_id in [0]:#range(9):
        #     for fib in range(trace_loc.shape[0]):
        #         plt.plot(trace_loc[fib, order_id, :], label="fibers", color="cyan",linestyle="--",linewidth=3)
        #     for fib in np.arange(0,trace_loc_slit.shape[0]):
        #         plt.plot(trace_loc_slit[fib, order_id, :], label="background", color="red",linestyle="-.",linewidth=3)
        #     for fib in np.arange(0,trace_loc_dark.shape[0]):
        #         plt.plot(trace_loc_dark[fib, order_id, :], label="dark", color="white",linestyle=":",linewidth=3)
        # plt.show()

        # plt.imshow(im_skysub*badpixmap,interpolation="nearest",origin="lower")
        # plt.show()

        fluxes, errors, residuals = extract_flux(im_skysub, badpixmap, line_width, trace_loc,fitbackground=fitbackground,bad_pixel_fraction=bad_pixel_fraction)
        fluxes_slit, _, _ = extract_flux(im_skysub, badpixmap, line_width_slit, trace_loc_slit,fitbackground=fitbackground,bad_pixel_fraction=bad_pixel_fraction)
        fluxes_dark, _, _ = extract_flux(im_skysub, badpixmap, line_width_dark, trace_loc_dark,fitbackground=fitbackground,bad_pixel_fraction=bad_pixel_fraction)

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=fluxes,header=header))
        hdulist.append(pyfits.ImageHDU(data=errors))
        hdulist.append(pyfits.ImageHDU(data=fluxes_slit))
        hdulist.append(pyfits.ImageHDU(data=fluxes_dark))
        out = os.path.join(mydir, os.path.basename(filename).replace(".fits","_fluxes.fits"))
        print("saving "+ out)
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=residuals,header=header))
        out = os.path.join(mydir, os.path.basename(filename).replace(".fits","_residuals.fits"))
        print("saving "+ out)
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()


        for fib in range(trace_loc.shape[0]):
            plt.figure(1+fib)
            for order_id in range(9):
                plt.subplot(9, 1, 9-order_id)
                plt.plot(fluxes[fib,order_id,:],linestyle="-",linewidth=2,label="data")
                print(fib, order_id, np.nanmedian(fluxes[fib,order_id,:]))
                # plt.plot(fluxes_slit[4,order_id,:],linestyle="-",linewidth=2,label="slit")
                # plt.plot(fluxes_dark[4,order_id,:],linestyle="-",linewidth=2,label="dark")
                # plt.plot(errors[0,order_id,:],linestyle="--",linewidth=2)
                # plt.plot(errors[1,order_id,:],linestyle=":",linewidth=2)
                # plt.plot(errors[2,order_id,:],linestyle="-.",linewidth=2)
                plt.legend()

            # plt.figure(10+fib)
            # plt.imshow(residuals,interpolation="nearest",origin="lower")
            # plt.clim(-0,100)
    plt.show()
    print(fluxes.shape)

    exit()

    # 0
    # 0
    # 3.5159992895346335
    # 0
    # 1
    # 11.093787635312234
    # 0
    # 2
    # 7.525693332239948
    # 0
    # 3 - 0.08290173345702663
    # 0
    # 4
    # 1.595002568976095
    # 0
    # 5
    # 4.21600911388275
    # 0
    # 6
    # 6.746910013328291
    # 0
    # 7
    # 3.4864455417155167
    # 0
    # 8 - 1.2369212087594692
    # 1
    # 0
    # 11.204628585686706
    # 1
    # 1
    # 22.00375937070628
    # 1
    # 2
    # 24.430018537543386
    # 1
    # 3
    # 25.506294856266607
    # 1
    # 4
    # 35.67577611554986
    # 1
    # 5
    # 29.474959577914422
    # 1
    # 6
    # 27.04426405384747
    # 1
    # 7
    # 24.070858112977803
    # 1
    # 8
    # 19.683601185186784
    # 2
    # 0
    # 1667.046856559848
    # 2
    # 1
    # 2145.577712249943
    # 2
    # 2
    # 3202.453599508626
    # 2
    # 3
    # 4073.7809161153864
    # 2
    # 4
    # 3658.9197510669433
    # 2
    # 5
    # 3651.376575624273
    # 2
    # 6
    # 3141.478555991807
    # 2
    # 7
    # 3023.2941017624635
    # 2
    # 8
    # 2569.262756435577