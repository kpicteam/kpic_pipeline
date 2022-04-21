import os
import glob
import kpicdrp.data as data
import kpicdrp.throughput as throughput


## Change local directory
kpicpublicdir = "../../public_kpic_data/" # main data dir

## Path relative to the public kpic directory
filelist_spectra = glob.glob(os.path.join(kpicpublicdir, "20200928_HIP_95771","fluxes", "*bkgdsub_spectra.fits"))

filename_oldwvs = os.path.join(kpicpublicdir, "utils", "first_guess_wvs_20200928_HIP_81497.fits")

# load wavelength solution
wvsoln = data.Wavecal(filepath=filename_oldwvs)

# load one spectrum to use to compute throughput
spectrum = data.Spectrum(filepath=filelist_spectra[0])
spectrum.calibrate_wvs(wvsoln) # make sure it is wavelength calibrated

# Star parameters
kmag = 0.52 # vega mag
teff = 3800 # Kelvin, rough estimate

peak_throughput = throughput.calculate_throughput(spectrum, kmag, bb_temp=teff)

print(peak_throughput)

