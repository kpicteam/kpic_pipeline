# EXAMPLE:
# JX: compute master background and bad pixel map from background images

from glob import glob
import os
from astropy.io import fits
from kpicdrp import background
import matplotlib.pyplot as plt
import numpy as np

# For multiple tint and coadds in one folder
main_dir = "../../kpic_analysis/tutorial_data/" # main data dir
bkgddir = os.path.join(main_dir,"20200702_backgrounds/") # background frames
filelist = glob(os.path.join(bkgddir,"raw","*.fits"))

# For multiple tint and coadds in one folder
# It will save the master backgrounds and bap pixel maps to bkgddir.
master_bkgds,badpixmaps,\
smoothed_thermal_noises,unique_tint,unique_coadds = background.process_backgrounds(filelist,save_loc=bkgddir)

# Plot the resulting master backgrounds and bad pixels
plt.figure(1)
for k,(background_med,badpixmap, tint,coadd) in enumerate(zip(master_bkgds,badpixmaps,unique_tint,unique_coadds)):
    print(k)
    plt.subplot(2,len(master_bkgds),k+1)
    plt.title("tint={0} coadd={1}".format(tint,coadd))
    plt.imshow(background_med, interpolation="nearest", origin="lower")
    med_val = np.nanmedian(background_med)
    plt.clim([0,2*med_val])

    plt.subplot(2,len(master_bkgds),k+len(master_bkgds)+1)
    plt.imshow(badpixmap, interpolation="nearest", origin="lower")
plt.subplot(2,len(master_bkgds),1)
plt.ylabel("Master backgrounds")
plt.subplot(2,len(master_bkgds),len(master_bkgds)+1)
plt.ylabel("Bad pix maps")
plt.show()

# OR

# For a single tint/number of coadds
# master_bkgd, smoothed_thermal_noise, badpixmap = background.make_badpixmap(filelist,plot=True) # does not save automatically
# background.save_bkgd_badpix(bkgddir,master_bkgd,badpixmap,smoothed_thermal_noise,header=fits.getheader(filelist[0]),
#                             readnoisebar=False)
