# EXAMPLE:
# JX: compute master background and bad pixel map from background images

from glob import glob
import os
from astropy.io import fits
from kpicdrp import background

# For multiple tint and coadds in one folder
main_dir = "../../kpic_analysis/tutorial_data/" # main data dir
bkgddir = os.path.join(main_dir,"20200702_backgrounds/") # background frames
filelist = glob(os.path.join(bkgddir,"raw","*.fits"))

# For multiple tint and coadds in one folder
master_bkgds,badpixmaps,\
smoothed_thermal_noises,unique_tint,unique_coadds = background.process_backgrounds(filelist,
                                                                                   plot=True,save_loc=bkgddir)
# OR

# For a single tint/number of coadds
# master_bkgd, smoothed_thermal_noise, badpixmap = background.make_badpixmap(filelist,plot=True) # does not save automatically
# background.save_bkgd_badpix(bkgddir,master_bkgd,badpixmap,smoothed_thermal_noise,header=fits.getheader(filelist[0]),
#                             readnoisebar=False)
