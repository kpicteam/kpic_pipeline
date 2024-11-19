# EXAMPLE:
# JX: compute master background and bad pixel map from background images
from glob import glob
import os
import sys
from kpicdrp import background
import matplotlib.pyplot as plt
import numpy as np
from kpicdrp.data import Dataset, DetectorFrame
from kpicdrp.caldb import det_caldb
from pipeline_utils import get_filenums, get_filelist

obsdate = input("Enter UT Date (e.g.20220723) >>> ")
obsdate = obsdate.strip()
print(obsdate)
main_calibdir = os.path.join("/scr3/kpic/KPIC_Campaign/calibs/", obsdate)

raw_datadir = os.path.join("/scr3/kpic/Data/", obsdate[2:], "spec")
print(raw_datadir)

filestr = "nspec"+obsdate[2:]+"_{0:04d}.fits"
filestr_alt = "nspec"+ str(int(obsdate[2:])+1) +"_{0:04d}.fits"
print(filestr, filestr_alt)

bkgddir = os.path.join(main_calibdir, "bkgd_bpmap/") # main data dir
if not os.path.exists(bkgddir):
    os.makedirs(bkgddir)

# check if raw directory exists. If not, create it and pull files over
raw_dir = os.path.join(bkgddir,"raw")
if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

# read calib config file
sys.path.insert(1, main_calibdir)
from calib_info import calib_info
print('Loaded calibration config file for ' + obsdate)
filenums = get_filenums(calib_info['raw_bkgd_range'])
# Only copy files if there are less files in raw dir than bkgd filenums
# figure out which are raw frame nums

## Hack up here ##
# If passing filestr_alt, it will try both
filelist = get_filelist(raw_dir, filenums, filestr, raw_datadir, filestr_alt=filestr_alt)

# if previous step failed, just grab all files in raw
if len(filelist) == 0:
    filelist = glob(os.path.join(raw_dir,"*.fits"))

print(filelist)
# read in the dataset
raw_dataset = Dataset(filelist=filelist, dtype=DetectorFrame)

# For multiple tint and coadds in one folder
# It will save the master backgrounds and bad pixel maps to bkgddir.
master_bkgds, badpixmaps, unique_tint, unique_coadds = background.process_backgrounds(raw_dataset, save_loc=bkgddir, fileprefix=obsdate, caldb_save_loc=det_caldb)

# Plot the resulting master backgrounds and bad pixels
plt.figure(1, figsize=(10,10))
for k,(background_med,badpixmap, tint,coadd) in enumerate(zip(master_bkgds,badpixmaps,unique_tint,unique_coadds)):
    print(k)
    plt.subplot(2,len(master_bkgds),k+1)
    plt.title("tint={0} coadd={1}".format(tint,coadd))
    plt.imshow(background_med.data, interpolation="nearest", origin="lower")
    med_val = np.nanmedian(background_med.data)
    plt.clim([0,2*med_val])

    plt.subplot(2,len(master_bkgds),k+len(master_bkgds)+1)
    plt.imshow(badpixmap.data, interpolation="nearest", origin="lower")
plt.subplot(2,len(master_bkgds),1)
plt.ylabel("Master backgrounds")
plt.subplot(2,len(master_bkgds),len(master_bkgds)+1)
plt.ylabel("Bad pix maps")
plt.savefig( os.path.join(bkgddir, 'bkgd_bpmap.png'), dpi=200 )
plt.show()
