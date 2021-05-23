# EXAMPLE:
# JX: compute master background and bad pixel map from background images

from glob import glob
import os
from astropy.io import fits
import kpicdrp.background as background
from kpicdrp.data import Dataset, DetectorFrame
import matplotlib.pyplot as plt
import numpy as np

# For multiple tint and coadds in one folder
kpicpublicdir = "fill/in/your/path/public_kpic_data/" # main data dir
kpicpublicdir = "../public_kpic_data" # main data dir
bkgddir = os.path.join(kpicpublicdir,"20200928_backgrounds") # background frames
save_loc =os.path.join(bkgddir,"calib")
# Create output directory if it does not exist.
if not os.path.exists(os.path.join(bkgddir,"calib")):
    os.makedirs(os.path.join(bkgddir,"calib"))
bkgd_fileprefix = "20200928"

# make a list of the input data
filelist = glob(os.path.join(bkgddir,"raw","*.fits"))

# read in the dataset
raw_dataset = Dataset(filelist=filelist, dtype=DetectorFrame)

# For multiple tint and coadds in one folder
# It will save the master backgrounds and bad pixel maps to bkgddir.
master_bkgds, badpixmaps, unique_tint, unique_coadds = background.process_backgrounds(raw_dataset, save_loc=save_loc, fileprefix=bkgd_fileprefix)

# Plot the resulting master backgrounds and bad pixels
plt.figure(1)
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
plt.show()

# OR

# For a single tint/number of coadds
# master_bkgd, smoothed_thermal_noise, badpixmap = background.create_background_badpixelmap(filelist,plot=True) # does not save automatically
# background.save_bkgd_badpix(bkgddir,master_bkgd,badpixmap,smoothed_thermal_noise,header=fits.getheader(filelist[0]),
#                             readnoisebar=False)
