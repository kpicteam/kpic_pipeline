
from kpicdrp.trace import *
import os
import multiprocessing as mp
from glob import glob
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

if __name__ == "__main__":

    try:
        import mkl

        mkl.set_num_threads(1)
    except:
        pass

    ## Change local directory
    kpicpublicdir = "/scr3/jruffio/data/kpic/public_kpic_data"
    obj_folder = "20200702_ups_Her"

    run_trace_fit(kpicpublicdir,obj_folder,N_order=9,usershift=0,make_guess=True)

    if 1:  # plot
        trace_loc_filename = glob(os.path.join(kpicpublicdir,obj_folder, "calib", "*_trace_loc_smooth.fits"))[0]
        hdulist = pyfits.open(trace_loc_filename)
        trace_loc = hdulist[0].data
        trace_loc[np.where(trace_loc == 0)] = np.nan
        print(trace_loc.shape)
        plt.figure(1)
        for order_id in range(9):
            plt.subplot(9, 1, 9-order_id)
            plt.plot(trace_loc[3,order_id,:],linestyle="-",linewidth=2)
        plt.show()

        trace_loc_slit = np.zeros((trace_loc.shape[0], trace_loc.shape[1], trace_loc.shape[2]))
        trace_loc_dark = np.zeros((trace_loc.shape[0] * 2, trace_loc.shape[1], trace_loc.shape[2]))
        for order_id in range(9):
            dy1 = np.nanmean(trace_loc[0, order_id, :] - trace_loc[1, order_id, :]) / 2
            dy2 = np.nanmean(trace_loc[0, order_id, :] - trace_loc[3, order_id, :])
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

        plt.figure(1)
        for order_id in range(9):
            for fib in range(trace_loc.shape[0]):
                plt.plot(trace_loc[fib, order_id, :], label="fibers", color="cyan", linestyle="--", linewidth=1)
            plt.plot(trace_loc[0, order_id, :], label="fibers", color="cyan", linestyle="-", linewidth=2)
            for fib in np.arange(0, trace_loc_slit.shape[0]):
                plt.plot(trace_loc_slit[fib, order_id, :], label="background", color="red", linestyle="-.", linewidth=1)
            plt.plot(trace_loc_slit[0, order_id, :], label="background", color="red", linestyle="-", linewidth=1)
            for fib in np.arange(0, trace_loc_dark.shape[0]):
                plt.plot(trace_loc_dark[fib, order_id, :], label="dark", color="black", linestyle=":", linewidth=2)
            plt.plot(trace_loc_dark[0, order_id, :], label="dark", color="black", linestyle="-", linewidth=2)
        plt.show()