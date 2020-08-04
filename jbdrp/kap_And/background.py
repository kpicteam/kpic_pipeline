import astropy.io.fits as pyfits
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import median_filter
from astropy.stats import mad_std
from scipy.signal import correlate2d
from copy import copy

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
    if 1:
        tint = 60
        # filename = os.path.join("/scr3/kpic/Data/191107/nspec","nspec191107_{0:04d}.fits".format(k))
        if tint == 600:
            background600_filelist = glob(os.path.join(kap_And_dir,"background_science","*.fits"))
        elif tint == 60:
            background600_filelist = glob(os.path.join(kap_And_dir,"background_star","*.fits"))

        background600_list = []
        for filename in background600_filelist:
            hdulist = pyfits.open(filename)
            background = hdulist[0].data.T
            background_header = hdulist[0].header
            hdulist.close()
            background600_list.append(background)
        background600_cube = np.array(background600_list)
        print(background600_cube.shape)
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=background600_cube))
        out = os.path.join(kap_And_dir,"calib","background{0}_cube.fits".format(tint))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()
        background600_med = np.nanmedian(background600_cube, axis=0)
        background600_cube_medsub = background600_cube-background600_med[None,:,:]

        background600_cube_medsub_std = mad_std(background600_cube_medsub[np.where(np.isfinite(background600_cube_medsub))])
        background600_cube_badpixmap = np.ones(background600_cube.shape)
        background600_cube_badpixmap[np.where(np.abs(background600_cube_medsub)>5*background600_cube_medsub_std)] = np.nan
        # plt.figure(1)
        # plt.subplot(1,2,1)
        # plt.imshow(background600_cube_medsub[0,:,:],interpolation="nearest")
        # plt.clim([0,30])
        # plt.subplot(1,2,2)
        # plt.imshow(background600_cube_medsub[0,:,:]*background600_cube_badpixmap[0,:,:],interpolation="nearest")
        # plt.clim([0,30])
        # plt.figure(2)
        # print(background600_cube_medsub_std)
        # hist, bin_edges = np.histogram(background600_cube_medsub[np.where(np.isfinite(background600_cube_badpixmap))],bins=1000)
        # plt.plot(bin_edges[1::],hist)
        # plt.show()

        background600_med = np.nanmedian(background600_cube*background600_cube_badpixmap, axis=0)
        background600_badpixmap = np.nanmedian(background600_cube_badpixmap, axis=0)
        background600_std = mad_std(background600_med[np.where(np.isfinite(background600_badpixmap))])
        print(background600_std)
        background600_badpixmap[np.where((np.abs(background600_med)>15*background600_std))] = np.nan
        ny,nx = background600_med.shape
        background600_med_tmp = copy(background600_med)
        background600_med_tmp[np.where(np.isnan(background600_badpixmap))] = np.nanmedian(background600_med)
        background600_med_lpf = correlate2d(background600_med_tmp, np.ones((100, 100)), mode="same")
        background600_med_hpf = background600_med - background600_med_lpf*np.nansum(background600_med_lpf*background600_med)/np.nansum(background600_med_lpf**2)
        background600_badpixmap[np.where(np.abs(background600_med_hpf)>5*mad_std(background600_med_hpf))] = np.nan
        # exit()
        background600_badpixmap[np.where(np.isnan(correlate2d(background600_badpixmap,np.ones((3,3)),mode="same")))] = np.nan

        if "MCDS" == background_header["SAMPMODE"].strip():
            print("MCDS")
            for col in np.arange(129,2048,128):
                background600_badpixmap[col-1:col+2,:] = np.nan
        if "CDS" == background_header["SAMPMODE"].strip():
            print("CDS")
            for col in np.arange(129-64,2048,64):
                background600_badpixmap[col,:] = np.nan

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=background600_med))
        out = os.path.join(kap_And_dir,"calib","background{0}_med.fits".format(tint))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=background600_badpixmap))
        out = os.path.join(kap_And_dir,"calib","background{0}_badpixmap.fits".format(tint))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        # plt.figure(3)
        # plt.imshow(background600_med_hpf*background600_badpixmap)
        # plt.show()

        # plt.figure(1)
        # plt.subplot(1,2,1)
        # plt.imshow(background600_med*background600_badpixmap,interpolation="nearest",origin="lower")
        # plt.clim([0,300])
        # plt.subplot(1,2,2)
        # plt.imshow(background600_badpixmap,interpolation="nearest",origin="lower")
        #
        # # plt.figure(2)
        # # hist, bin_edges = np.histogram(background600_med*background600_badpixmap,bins=10000)
        # # plt.plot(bin_edges[1::],hist)
        # plt.show()
        # exit()
    else:
        background600_med_filename = os.path.join(kap_And_dir,"calib","background600_med.fits")
        hdulist = pyfits.open(background600_med_filename)
        background600_med = hdulist[0].data
        background600_badpixmap_filename = os.path.join(kap_And_dir,"calib","background600_badpixmap.fits")
        hdulist = pyfits.open(background600_badpixmap_filename)
        background600_badpixmap = hdulist[0].data
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(background600_med*background600_badpixmap,interpolation="nearest",origin="lower")
        plt.subplot(1,2,2)
        plt.imshow(background600_badpixmap,interpolation="nearest",origin="lower")
        plt.show()
