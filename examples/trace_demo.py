
import os
import multiprocessing as mp
from glob import glob
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import kpicdrp.trace as trace
import astropy.io.fits as pyfits

if __name__ == "__main__":

    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    # Public KPIC google drive
    # kpicpublicdir = "/scr3/jruffio/data/kpic/public_kpic_data"
    kpicpublicdir = ""

    # master background filelanme. make sure it is right integration time and coadd
    # background_med_filename = glob(os.path.join(kpicpublicdir,"20200702_ups_Her","calib","*background*.fits"))[0]
    background_med_filename = ""

    # Bad pixel map filename
    # persisbadpixmap_filename = glob(os.path.join(kpicpublicdir,"20200702_ups_Her","calib","*persistent_badpix*.fits"))[0]
    persisbadpixmap_filename = ""

    # List of raw files for the derivation of the trace
    # raw_data_dir = os.path.join(kpicpublicdir,"20200702_ups_Her","raw") # the star of interest
    raw_data_dir = ""

    filelist = glob(os.path.join(raw_data_dir, "*.fits"))
    # Set output directory
    # outdir = os.path.join(kpicpublicdir,"20200702_ups_Her","calib")
    outdir = ""

    # If True, it automatically derives a first guess for the traces location using kpicdrp.trace.fibers_guess() using
    # some peak fitting routine.
    make_guess = True
    N_order = 9
    # Set number of threads to be used
    numthreads = 16

    #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # read the master Background file
    hdulist = pyfits.open(background_med_filename)
    background = hdulist[0].data
    background_header = hdulist[0].header
    tint_bckg = int(background_header["ITIME"])

    # read the bad pixel map
    hdulist = pyfits.open(persisbadpixmap_filename)
    badpixmap = hdulist[0].data
    ny,nx = badpixmap.shape

    # Read the raw detector images into a cube while subtracting background
    im_list = []
    for filename in filelist:
        hdulist = pyfits.open(filename)
        im = hdulist[0].data.T[:,::-1]
        header = hdulist[0].header
        if tint_bckg != int(header["ITIME"]):
            raise Exception("bad tint {0}, should be {1}: ".format(int(header["ITIME"]),tint_bckg) + filename)
        hdulist.close()
        im_skysub = im-background
        im_list.append(im_skysub)
    cube = np.array(im_list)
    badpixcube = np.tile(badpixmap[None,:,:],(len(filelist),1,1))

    # Define the first guess for the trace location
    if make_guess:
        fibers = trace.fibers_guess(cube-badpixcube,N_order=N_order)
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
    for image in cube-badpixcube:
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
    hdulist.append(pyfits.PrimaryHDU(data=trace_width,header=header))
    hdulist.append(pyfits.ImageHDU(data=trace_loc))
    outfilename = os.path.join(outdir,os.path.basename(filelist[0]).replace(".fits","_trace_loc.fits"))
    try:
        hdulist.writeto(outfilename, overwrite=True)
    except TypeError:
        hdulist.writeto(outfilename, clobber=True)
    hdulist.close()


    if 1:  # plot
        hdulist = pyfits.open(outfilename)
        trace_width = hdulist[0].data
        trace_loc = hdulist[1].data
        trace_loc[np.where(trace_loc == 0)] = np.nan
        print(trace_loc.shape)
        # plt.figure(1)
        # for order_id in range(9):
        #     plt.subplot(9, 1, 9-order_id)
        #     plt.plot(trace_loc[3,order_id,:],linestyle="-",linewidth=2)

        plt.figure(2)
        for order_id in range(9):
            for fib in range(trace_loc.shape[0]):
                plt.plot(trace_loc[fib, order_id, :], label="fibers", color="cyan", linestyle="--", linewidth=1)
            plt.plot(trace_loc[0, order_id, :], label="fibers", color="cyan", linestyle="-", linewidth=2)
        plt.show()

    # if 1:  # plot
    #     trace_loc_filename = glob(os.path.join(kpicpublicdir,obj_folder, "calib", "*_trace_loc_smooth.fits"))[0]
    #     hdulist = pyfits.open(trace_loc_filename)
    #     trace_loc = hdulist[0].data
    #     trace_loc[np.where(trace_loc == 0)] = np.nan
    #     print(trace_loc.shape)
    #     plt.figure(1)
    #     for order_id in range(9):
    #         plt.subplot(9, 1, 9-order_id)
    #         plt.plot(trace_loc[3,order_id,:],linestyle="-",linewidth=2)
    #     plt.show()
    #
    #     trace_loc_slit = np.zeros((trace_loc.shape[0], trace_loc.shape[1], trace_loc.shape[2]))
    #     trace_loc_dark = np.zeros((trace_loc.shape[0] * 2, trace_loc.shape[1], trace_loc.shape[2]))
    #     for order_id in range(9):
    #         dy1 = np.nanmean(trace_loc[0, order_id, :] - trace_loc[1, order_id, :]) / 2
    #         dy2 = np.nanmean(trace_loc[0, order_id, :] - trace_loc[3, order_id, :])
    #         # exit()
    #         if np.isnan(dy1):
    #             dy1 = 10
    #         if np.isnan(dy2):
    #             dy2 = 40
    #         print(dy1, dy2)
    #
    #         trace_loc_slit[0, order_id, :] = trace_loc[0, order_id, :] - dy1
    #         trace_loc_slit[1, order_id, :] = trace_loc[1, order_id, :] - dy1
    #         trace_loc_slit[2, order_id, :] = trace_loc[2, order_id, :] - dy1
    #         trace_loc_slit[3, order_id, :] = trace_loc[3, order_id, :] - dy1
    #
    #         if order_id == 0:
    #             trace_loc_dark[0, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 0 * dy1
    #             trace_loc_dark[1, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 1 * dy1
    #             trace_loc_dark[2, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 2 * dy1
    #             trace_loc_dark[3, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 3 * dy1
    #
    #             trace_loc_dark[4, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 4 * dy1
    #             trace_loc_dark[5, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 5 * dy1
    #             trace_loc_dark[6, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 6 * dy1
    #             trace_loc_dark[7, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 7 * dy1
    #         else:
    #             trace_loc_dark[0, order_id, :] = trace_loc[0, order_id, :] - 3 * dy2 + 3 * dy1
    #             trace_loc_dark[1, order_id, :] = trace_loc[1, order_id, :] - 3 * dy2 + 4 * dy1
    #             trace_loc_dark[2, order_id, :] = trace_loc[2, order_id, :] - 3 * dy2 + 5 * dy1
    #             trace_loc_dark[3, order_id, :] = trace_loc[3, order_id, :] - 3 * dy2 + 6 * dy1
    #
    #             trace_loc_dark[4, order_id, :] = trace_loc[0, order_id, :] - 2 * dy2 + 2 * dy1
    #             trace_loc_dark[5, order_id, :] = trace_loc[1, order_id, :] - 2 * dy2 + 3 * dy1
    #             trace_loc_dark[6, order_id, :] = trace_loc[2, order_id, :] - 2 * dy2 + 4 * dy1
    #             trace_loc_dark[7, order_id, :] = trace_loc[3, order_id, :] - 2 * dy2 + 5 * dy1
    #
    #     plt.figure(1)
    #     for order_id in range(9):
    #         for fib in range(trace_loc.shape[0]):
    #             plt.plot(trace_loc[fib, order_id, :], label="fibers", color="cyan", linestyle="--", linewidth=1)
    #         plt.plot(trace_loc[0, order_id, :], label="fibers", color="cyan", linestyle="-", linewidth=2)
    #         for fib in np.arange(0, trace_loc_slit.shape[0]):
    #             plt.plot(trace_loc_slit[fib, order_id, :], label="background", color="red", linestyle="-.", linewidth=1)
    #         plt.plot(trace_loc_slit[0, order_id, :], label="background", color="red", linestyle="-", linewidth=1)
    #         for fib in np.arange(0, trace_loc_dark.shape[0]):
    #             plt.plot(trace_loc_dark[fib, order_id, :], label="dark", color="black", linestyle=":", linewidth=2)
    #         plt.plot(trace_loc_dark[0, order_id, :], label="dark", color="black", linestyle="-", linewidth=2)
    #     plt.show()