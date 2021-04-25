
import os
import multiprocessing as mp
from glob import glob
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import kpicdrp.trace as trace
import astropy.io.fits as pyfits
import kpicdrp.extraction as extraction
import kpicdrp.utils as utils

if __name__ == "__main__":

    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    # Public KPIC google drive
    kpicpublicdir = "fill/in/your/path/public_kpic_data/" # main data dir

    # master background and bad pix directory.
    mybkgdir = os.path.join(kpicpublicdir,"20200928_backgrounds","calib")
    # mybkgdir = ""
    background_med_filename = os.path.join(mybkgdir,"20200928_background_med_nobars_tint4.42584_coadds1.fits")
    persisbadpixmap_filename = os.path.join(mybkgdir,"20200928_persistent_badpix_nobars_tint4.42584_coadds1.fits")

    # List of raw files for the derivation of the trace
    raw_data_dir = os.path.join(kpicpublicdir,"20200928_zet_Aql","raw") # the star of interest
    # raw_data_dir = ""
    filelist = glob(os.path.join(raw_data_dir, "*.fits"))

    # Set output directory
    out_trace_dir = os.path.join(kpicpublicdir,"20200928_zet_Aql","calib")
    outfilename = os.path.join(out_trace_dir,os.path.basename(filelist[0]).replace(".fits","_trace.fits"))
    # out_trace_dir = ""
    if not os.path.exists(os.path.join(out_trace_dir)):
        os.makedirs(os.path.join(out_trace_dir))

    # If True, it automatically derives a first guess for the traces location using kpicdrp.trace.fibers_guess() using
    # some peak fitting routine.
    make_guess = True
    N_order = 9
    # Set number of threads to be used
    numthreads = 16

    #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # read the master Background file
    # bkgd, bkgd_noise, badpixmap = utils.get_calib_bkg(filelist[0], mybkgdir)
    # Or manually....
    # read the master Background file
    _hdulist = pyfits.open(background_med_filename)
    bkgd = _hdulist[0].data
    bkgd_noise = _hdulist[1].data
    # read the bad pixel map
    _hdulist = pyfits.open(persisbadpixmap_filename)
    badpixmap = _hdulist[0].data
    ny,nx = badpixmap.shape

    badpixcube = np.tile(badpixmap[None,:,:],(len(filelist),1,1))

    # Read the raw detector images into a cube while subtracting background
    sci_frames,sci_hdrs = extraction.process_sci_raw2d(filelist, bkgd, badpixmap, detect_cosmics=True, scale=False,add_baryrv=False)
    cube = np.array(sci_frames)

    # Define the first guess for the trace location
    if make_guess:
        fibers = trace.fibers_guess(cube*badpixcube,N_order=N_order)
    else:
        fiber1 = [[70, 150], [260, 330], [460, 520], [680 - 10, 720 + 10], [900 - 15, 930 + 15], [1120 - 5, 1170 + 5],
                  [1350, 1420], [1600, 1690], [1870, 1980]]
        fiber2 = [[50, 133], [240, 320], [440, 510], [650, 710], [880 - 15, 910 + 15], [1100 - 5, 1150 + 5], [1330, 1400],
                  [1580, 1670], [1850, 1960]]
        fiber3 = [[30, 120], [220, 300], [420, 490], [640 - 5, 690 + 5], [865 - 20, 890 + 20], [1090 - 10, 1130 + 10],
                  [1320, 1380], [1570, 1650], [1840, 1940]]
        fibers = {0: fiber1, 1: fiber2, 2: fiber3}

    # Identify which fibers was observed in each file
    fiber_list = []
    for image in cube*badpixcube:
        fiber_list.append(trace.guess_star_fiber(image,fibers))

    # Calibrate the trace position and width
    trace_calib,residuals = trace.fit_trace(fibers,fiber_list,cube,badpixcube,ny,nx,N_order,numthreads=numthreads,fitbackground=False)
    # The dimensions of trace calib are (4 fibers, 9 orders, 2048 pixels, 5) #[A, w, y0, rn, B]
    # trace_calib[:,:,:,0]: amplitude of the 1D gaussian
    # trace_calib[:,:,:,1]: trace width (1D gaussian sigma)
    # trace_calib[:,:,:,2]: trace y-position
    # trace_calib[:,:,:,3]: noise (ignore)
    # trace_calib[:,:,:,4]: background (ignore)

    # Smooth the trace calibrations different ways, with polyfit or with spline. Only using the spline smoothing.
    _,smooth_trace_calib = trace.smooth(trace_calib)

    trace_width = smooth_trace_calib[:,:,:,1]
    trace_loc = smooth_trace_calib[:,:,:,2]

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=trace_width,header=sci_hdrs[0]))
    hdulist.append(pyfits.ImageHDU(data=trace_loc))
    try:
        hdulist.writeto(outfilename, overwrite=True)
    except TypeError:
        hdulist.writeto(outfilename, clobber=True)
    hdulist.close()


    if 1:  # plot
        hdulist = pyfits.open(outfilename)
        trace_width = hdulist[0].data
        trace_loc = hdulist[1].data

        trace_loc_slit,trace_loc_dark = trace.get_background_traces(trace_loc)

        plt.figure(2)
        for order_id in range(9):
            for fib in range(trace_loc.shape[0]):
                plt.plot(trace_loc[fib, order_id, :], label="fibers", color="cyan", linestyle="-", linewidth=1)
                plt.plot(trace_loc_slit[fib, order_id, :], label="fibers", color="grey", linestyle="--", linewidth=1)
                plt.plot(trace_loc_dark[fib, order_id, :], label="fibers", color="black", linestyle="--", linewidth=1)
        plt.show()