# EXAMPLE:

from glob import glob
import os
from astropy.io import fits
from kpicdrp import background

# For multiple tint and coadds in one folder
mykpicdir = "../kpic/"
bkgddir = os.path.join(mykpicdir,"20200702_backgrounds")
filelist = glob(os.path.join(mydir,"raw","*.fits"))

# For multiple tint and coadds in one folder
master_bkgds,badpixmaps,smoothed_thermal_noises,unique_tint,unique_coadds = backgrounds.process_backgrounds(filelist,plot=False,save_loc=bkgddir)

# OR

# For a single tint/number of coadds
master_bkgd, smoothed_thermal_noise, badpixmap = backgrounds.make_badpixmap(filelist,plot=False) # does not save automatically
save_bkgd_badpix(master_bkgd,badpixmap,smoothed_thermal_noise,header=fits.getheader(filelist[0]),readnoisebar=False)

