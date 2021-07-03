import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.table as table
import scipy.ndimage as ndi
import os
import multiprocessing as mp
from glob import glob
import kpicdrp.utils as utils
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import kpicdrp.trace as trace
import kpicdrp.extraction as extraction

if __name__ == "__main__":

    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    raw_folder = "/scr3/kpic/Data/210425/spec/"
    out_folder = "/scr3/jruffio/data/kpic/20210425_LSRJ1835+3259/raw_pairsub/"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # filenums_fib2 = [321,323,325,326,327,328,333,335,336,339]
    # filenums_fib3 = [322,324,329,330,331,332,334,337,338,340]

    init_num, Nim, Nit = 84,1,10
    filenums_fib2 = []
    filenums_fib3 = []
    currit = init_num
    for k in range(Nit):
        for l in range(Nim):
            filenums_fib2.append(currit)
            currit += 1
        for l in range(Nim):
            filenums_fib3.append(currit)
            currit += 1
    print(filenums_fib2)
    print(filenums_fib3)
    # exit()

    for filenum2,filenum3 in zip(filenums_fib2,filenums_fib3):
        filename2 = os.path.join(raw_folder,"nspec210425_{0:04d}.fits".format(filenum2))
        filename3 = os.path.join(raw_folder,"nspec210425_{0:04d}.fits".format(filenum3))
        hdulist = fits.open(filename2)
        im2 = hdulist[0].data
        hdr2 = hdulist[0].header
        hdulist = fits.open(filename3)
        im3 = hdulist[0].data
        hdr3 = hdulist[0].header

        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=im2-im3, header=hdr2))
        out = os.path.join(out_folder,os.path.basename(filename2).replace(".fits","_pairsub.fits"))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=im3-im2, header=hdr3))
        out = os.path.join(out_folder,os.path.basename(filename3).replace(".fits","_pairsub.fits"))
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        # exit()