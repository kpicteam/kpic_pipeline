import os
import numpy as np
import astropy.io.fits as fits
import kpicdrp.data as data
from kpicdrp.caldb import det_caldb
import kpicdrp.extraction as extraction

from glob import glob

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
goal_fibers = []
currit = init_num
for k in range(Nit):
    for l in range(Nim):
        goal_fibers += [2, ]
    for l in range(Nim):
        goal_fibers += [3, ]
print(goal_fibers)
# exit()

template_fname = "nspec210425_{0:04d}.fits"
filenums = range(init_num, init_num + (Nim * Nit * 2))
filelist = [os.path.join(raw_folder, template_fname.format(i)) for i in filenums]

raw_sci_dataset = data.Dataset(filelist=filelist, dtype=data.DetectorFrame)

# fetch calibration files
badpixmap = det_caldb.get_calib(raw_sci_dataset[0], type="BadPixelMap")

sci_dataset = extraction.process_sci_raw2d(raw_sci_dataset, None, badpixmap, detect_cosmics=True, add_baryrv=True, nod_subtraction='pair', fiber_goals=goal_fibers)

sci_dataset.save(filedir=out_folder)
