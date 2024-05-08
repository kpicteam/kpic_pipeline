# JX: compute trace location and widths from a standard star (A or B type ideally)
import os
from glob import glob
import numpy as np
import sys
import matplotlib.pyplot as plt
from glob import glob
import kpicdrp.trace as trace
import kpicdrp.extraction as extraction
import kpicdrp.trace as trace
import kpicdrp.extraction as extraction
import kpicdrp.utils as utils
import kpicdrp.data as data
from kpicdrp.caldb import det_caldb, trace_caldb
from pipeline_utils import get_filenums, get_filelist

try:
    import mkl
    mkl.set_num_threads(12)
except:
    pass

obsdate = input("Enter UT Date (e.g.20220723) >>> ")
obsdate = obsdate.strip()
print(obsdate)

main_calibdir = os.path.join("/scr3/kpic/KPIC_Campaign/calibs/", obsdate)
raw_datadir = os.path.join("/scr3/kpic/Data/", obsdate[2:], "spec")
filestr = "nspec"+obsdate[2:]+"_0{0:03d}.fits"

# read calib config file
sys.path.insert(1, main_calibdir)
from calib_info import calib_info
print('Loaded calibration config file for ' + obsdate)
trace_star = calib_info['trace_star']

# Figure out raw files for the derivation of the trace
filenums = get_filenums(calib_info['raw_trace_range'])

trace_star_dir = os.path.join("/scr3/kpic/KPIC_Campaign/science", trace_star, obsdate)
# check if raw directory exists. If not, create it and pull files over
raw_dir = os.path.join(trace_star_dir, "raw")
if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

# Only copy files if there are less files in raw dir than bkgd filenums
filelist = get_filelist(raw_dir, filenums, filestr, raw_datadir)

# Set output directory
out_trace_dir = os.path.join(main_calibdir, "trace")
if not os.path.exists(os.path.join(out_trace_dir)):
    os.makedirs(os.path.join(out_trace_dir))
# save starname used for the trace
outfilename = os.path.join(out_trace_dir, trace_star+"_trace.fits")

# If True, it automatically derives a first guess for the traces location using kpicdrp.trace.fibers_guess() using
# some peak fitting routine.
make_guess = True
N_order = 9
# Set number of threads to be used
numthreads = 12

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
    
    print('guess_labels')
    print(guess_labels)

    guess_locs = np.array(guess_locs)
    guess_labels = np.array(guess_labels)
    guess_widths = np.ones(guess_locs.shape) * (3 / (2 * np.sqrt(2 * np.log(2))))
    guess_fibers_params = data.TraceParams(locs=guess_locs, widths=guess_widths, labels=guess_labels, header=cleaned_data[0].header)
    
# Calibrate the trace position and width
trace_calib = trace.fit_trace(cleaned_data, guess_fibers_params, numthreads=numthreads, fitbackground=False)

# save the unsmoothed trace as well
trace_calib.save(filedir=out_trace_dir, filename=outfilename.replace('trace.fits', 'unsmoothed_trace.fits'))

# Smooth the trace calibrations different ways, with polyfit or with spline. Only using the spline smoothing.
smooth_trace_calib = trace.smooth(trace_calib)

# add background and dark current traces to sample the noise (optional)
smooth_trace_calib = trace.add_background_traces(smooth_trace_calib)

smooth_trace_calib.save(filedir=out_trace_dir, filename=outfilename, caldb=trace_caldb)

# Plot
trace_width = smooth_trace_calib.widths
trace_loc = smooth_trace_calib.locs

plt.figure(2)
for order_id in range(9):
    # only plot the trace of sci fibers
    for fib in range(4):
        label = smooth_trace_calib.labels[fib]
        if '1' in label:
            color = 'blue'
        elif '2' in label:
            color = 'red'
        elif '3' in label:
            color = 'black'
        elif '4' in label:
            color = 'green'
        linestyle = '--'
        if order_id == 0:
            plt.plot(trace_loc[fib, order_id, :], label=label, color=color, linestyle=linestyle, linewidth=1)
        else:
            plt.plot(trace_loc[fib, order_id, :], label='__nolegend__', color=color, linestyle=linestyle, linewidth=1)

plt.legend()
plt.savefig( os.path.join(out_trace_dir, trace_star+'_trace.png'), dpi=200 )
plt.show()
