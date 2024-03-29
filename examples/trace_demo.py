
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
import kpicdrp.data as data
from kpicdrp.caldb import det_caldb, trace_caldb

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

# Public KPIC google drive
kpicpublicdir = "fill/in/your/path/public_kpic_data/" # main data dir

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
numthreads = 4

# read in the raw data
input_data = data.Dataset(filelist=filelist, dtype=data.DetectorFrame)

# master background and bad pix directory
bkgd = det_caldb.get_calib(input_data[0], type="Background")
badpixmap = det_caldb.get_calib(input_data[0], type="BadPixelMap")

# Read the raw detector images into a cube while subtracting background
cleaned_data = extraction.process_sci_raw2d(input_data, bkgd, badpixmap, detect_cosmics=True, add_baryrv=False)

# Define the first guess for the trace location
if make_guess:
    guess_fibers_params = trace.fibers_guess(cleaned_data, N_order=N_order)
else:
    fiber1 = [[70, 150], [260, 330], [460, 520], [680 - 10, 720 + 10], [900 - 15, 930 + 15], [1120 - 5, 1170 + 5],
                [1350, 1420], [1600, 1690], [1870, 1980]]
    fiber2 = [[50, 133], [240, 320], [440, 510], [650, 710], [880 - 15, 910 + 15], [1100 - 5, 1150 + 5], [1330, 1400],
                [1580, 1670], [1850, 1960]]
    fiber3 = [[30, 120], [220, 300], [420, 490], [640 - 5, 690 + 5], [865 - 20, 890 + 20], [1090 - 10, 1130 + 10],
                [1320, 1380], [1570, 1650], [1840, 1940]]

    guess_locs = []
    guess_labels = []
    for k,fiber_ends in enumerate([fiber1, fiber2, fiber3]):
        guess_locs_thisfib = []
        for ends in fiber_ends:
            num_channels = cleaned_data[0].shape[-1]
            this_order_guesspos = np.interp(np.arange(num_channels), [0, num_channels], ends)
            guess_locs_thisfib.append(this_order_guesspos)
        guess_locs.append(guess_locs_thisfib)
        guess_labels.append("s{0}".format(k + 1))

    guess_locs = np.array(guess_locs)
    guess_labels = np.array(guess_labels)
    guess_widths = np.ones(guess_locs.shape) * (3 / (2 * np.sqrt(2 * np.log(2))))
    guess_fibers_params = data.TraceParams(locs=guess_locs, widths=guess_widths, labels=guess_labels, header=cleaned_data[0].header)
    

# Calibrate the trace position and width
trace_calib = trace.fit_trace(cleaned_data, guess_fibers_params, numthreads=numthreads, fitbackground=False)

# Smooth the trace calibrations different ways, with polyfit or with spline. Only using the spline smoothing.
smooth_trace_calib = trace.smooth(trace_calib)

# add background and dark current traces to sample the noise (optional)
smooth_trace_calib = trace.add_background_traces(smooth_trace_calib)

smooth_trace_calib.save(filedir=out_trace_dir, filename=os.path.basename(filelist[0]).replace(".fits","_trace.fits"), caldb=trace_caldb)

if 1:  # plot
    # hdulist = pyfits.open(outfilename)
    # trace_width = hdulist[0].data
    # trace_loc = hdulist[1].data
    trace_width = smooth_trace_calib.widths
    trace_loc = smooth_trace_calib.locs

    # trace_loc_slit,trace_loc_dark = trace.get_background_traces(trace_loc)

    plt.figure(2)
    for order_id in range(9):
        for fib in range(trace_loc.shape[0]):
            label = smooth_trace_calib.labels[fib]
            if 's' in label:
                color = 'cyan'
                linestyle = '-'
            elif 'b' in label:
                color = 'grey'
                linestyle = "--"
            elif 'd' in label:
                color = 'black'
                linestyle = "--"
            plt.plot(trace_loc[fib, order_id, :], label="fibers", color=color, linestyle=linestyle, linewidth=1)

    plt.show()