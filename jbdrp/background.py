import astropy.io.fits as pyfits
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import median_filter
from astropy.stats import mad_std
from scipy.signal import correlate2d
from copy import copy
from scipy.interpolate import interp1d
from scipy.ndimage.filters import convolve
from utils.badpix import *

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    mykpicdir = "/scr3/jruffio/data/kpic/"
    # mydir = os.path.join(mykpicdir,"20191012_backgrounds")
    # mydir = os.path.join(mykpicdir,"20191013_backgrounds")
    # mydir = os.path.join(mykpicdir,"20191014_backgrounds")
    # mydir = os.path.join(mykpicdir,"20191107_backgrounds")
    # mydir = os.path.join(mykpicdir,"20191214_backgrounds")
    # mydir = os.path.join(mykpicdir,"20191215_backgrounds")
    # mydir = os.path.join(mykpicdir,"20200607_backgrounds")
    # mydir = os.path.join(mykpicdir,"20200608_backgrounds")
    # mydir = os.path.join(mykpicdir,"20200609_backgrounds")
    # mydir = os.path.join(mykpicdir,"20200701_backgrounds")
    mydir = os.path.join(mykpicdir,"20200702_backgrounds")
    readnoisebar = False



    mydate = os.path.basename(mydir).split("_")[0]

    filelist = glob(os.path.join(mydir,"raw","*.fits"))
    print(os.path.join(mydir,"raw","*.fits"))
    print(len(filelist))
    # exit()

    ## Read ITIME and temperature
    # print(filelist)
    tint_list = []
    coadds_list = []
    background_list = []
    background_badpix_list = []
    header_list = []
    for filename in filelist:
        print(filename)
        hdulist = pyfits.open(filename)
        background = hdulist[0].data.T[:,::-1]
        background_header = hdulist[0].header

        background_badpixmap = get_badpixmap_from_laplacian(background,bad_pixel_fraction=1e-2)
        background_badpixmap = background_badpixmap*get_badpixmap_from_mad(background,threshold=10)
        if readnoisebar:
            background_badpixmap = background_badpixmap*get_badpixmap_from_readnoisebars(background,background_header)

        # plt.figure(2)
        # plt.imshow(background*background_badpixmap,interpolation="nearest",origin="lower")
        # med_val = np.nanmedian(background)
        # plt.clim([0,2*med_val])
        # plt.show()

        tint_list.append(float(background_header["TRUITIME"]))
        coadds_list.append(int(background_header["COADDS"]))
        background_list.append(background*background_badpixmap)
        background_badpix_list.append(background_badpixmap)
        header_list.append(background_header)
        hdulist.close()

    unique_tint = np.unique(tint_list)
    unique_coadds = np.unique(coadds_list)
    # print(tint_list)
    for tint in unique_tint:
        for coadds in unique_coadds:
            print("tint={0}".format(tint))
            print("coadds={0}".format(coadds))
            where_tint = np.where((tint_list==tint)*(coadds_list==coadds))
            if np.size(where_tint[0]) == 0:
                continue
            print("N files = {0}".format(np.size(where_tint[0])))
            background_cube = np.array(background_list)[where_tint[0],:,:]
            background_badpix_cube = np.array(background_badpix_list)[where_tint[0],:,:]


            background_med = np.nanmedian(background_cube*background_badpix_cube, axis=0)

            persistent_badpix = np.ones(background_med.shape)
            persistent_badpix[np.where(np.nansum(background_badpix_cube,axis=0)<np.max([2,0.25*background_cube.shape[0]]))] = np.nan

            # plt.figure(2)
            # plt.imshow(background_med,interpolation="nearest",origin="lower")
            # med_val = np.nanmedian(background_med)
            # # plt.clim([0,2*med_val])
            # plt.show()

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=background_med,header=header_list[where_tint[0][0]]))
            if readnoisebar:
                out = os.path.join(mydir,mydate+"_background_med_tint{0}_coadds{1}.fits".format(tint,coadds))
            else:
                out = os.path.join(mydir,mydate+"_background_med_nobars_tint{0}_coadds{1}.fits".format(tint,coadds))
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=persistent_badpix,header=header_list[where_tint[0][0]]))
            if readnoisebar:
                out = os.path.join(mydir,mydate+"_persistent_badpix_tint{0}_coadds{1}.fits".format(tint,coadds))
            else:
                out = os.path.join(mydir,mydate+"_persistent_badpix_nobars_tint{0}_coadds{1}.fits".format(tint,coadds))
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()
