"""
X-Corr ROXs 42 Bb
"""
import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.table as table
import scipy.ndimage as ndi
import os
import multiprocessing as mp
import astropy.units as u

import matplotlib.pylab as plt

import kpicdrp.extraction
import kpicdrp.xcorr as xcorr


# Run the extraction code on the planet data

main_dir = "../../public_kpic_data/" # main data dir
target_dir = os.path.join(main_dir,"20200928_HR_7672B") # the star of interest
calib_dir = os.path.join(target_dir, "calib") # calib subdir
raw_data_dir = os.path.join(target_dir, "raw") # raw 2D images

# select science files
filestr = "nspec200928_0{0:03d}.fits"
filenums =  [77, 83]
filelist = [os.path.join(raw_data_dir, filestr.format(i)) for i in filenums]

# load wavelength solution
with fits.open(os.path.join(calib_dir, "20200928_HIP_95771_wvs.fits")) as hdulist:
    all_wvs = np.copy(hdulist[0].data)
    sf2_wvs = all_wvs[1]

# with fits.open(os.path.join("../../public_kpic_data/20200702_backgrounds/", "_background_med_nobars_tint598.963684_coadds1.fits")) as hdulist:
#     bkgd_600s = np.copy(hdulist[0].data)

# with fits.open(os.path.join("../../public_kpic_data/20200702_backgrounds/", "_persistent_badpix_nobars_tint598.963684_coadds1.fits")) as hdulist:
#     badpixmap_600s = np.copy(hdulist[0].data)

# # read in trace centers and trace widths
# # shape is [nfibers, norders, npixels]
# with fits.open(os.path.join(calib_dir, "20200702_trace_loc_smooth.fits")) as hdulist:
#     trace_centers = hdulist[0].data
# with fits.open(os.path.join(calib_dir, "20200702_line_width_smooth.fits")) as hdulist:
#     trace_sigmas = hdulist[0].data


# # read in the data and do basic processing
# nobadpix = np.zeros(bkgd_600s.shape) # can be used if you want to skip bad pixel map application
# mean_sci_data, sci_hdrs, sci_noise, sci_frames = kpicdrp.extraction.process_sci_raw2d(filelist, bkgd_600s,
#                                                         badpixmap_600s, detect_cosmics=True)
# # extraction
# # option of multiprocessing, arg is num processes - set pool to None if not desired
# pool = mp.Pool(2)

# for frame, num, hdr in zip(sci_frames, filenums, sci_hdrs):
#     fluxes, error = kpicdrp.extraction.extract_flux(frame,
#                                 os.path.join(target_dir + "nspec200702_0{0:03d}_fluxes.fits".format(num)),
#                                 trace_centers, trace_sigmas, img_hdr=hdr, fit_background=True, pool=pool)

all_fluxes = []
all_errors = []
for num in filenums:
    filepath = os.path.join(target_dir, "nspec200928_0{0:03d}_fluxes.fits".format(num))
    with fits.open(filepath) as hdulist:
        flux = hdulist[0].data
        err = hdulist[1].data

    all_fluxes.append(flux)
    all_errors.append(err)

all_fluxes = np.array(all_fluxes)
all_errors = np.array(all_errors)

print(np.nanmean(all_fluxes, axis=(1,2)))

# combine in time
tot_fluxes = np.nansum(all_fluxes, axis=0)
tot_errors = np.sqrt(np.nansum(all_errors**2, axis=0))

sf2_fluxes = tot_fluxes[1]
sf2_fluxes[:,:50] = np.nan
sf2_fluxes[:,-150:] = np.nan

# read in star fluxes

star_target_dir = os.path.join(main_dir,"20200928_HR_7672") # the star of interest
star_filenums =  [71, 84]

all_star_fluxes = []
all_star_errors = []
for num in star_filenums:
    filepath = os.path.join(star_target_dir, "nspec200928_0{0:03d}_fluxes.fits".format(num))
    with fits.open(filepath) as hdulist:
        flux = hdulist[0].data
        err = hdulist[1].data

    all_star_fluxes.append(flux)
    all_star_errors.append(err)

all_star_fluxes = np.array(all_star_fluxes)
all_star_errors = np.array(all_star_errors)

print(np.nanmean(all_star_fluxes, axis=(1,2)))

# combine in time
tot_star_fluxes = np.nansum(all_star_fluxes, axis=0)
tot_star_errors = np.sqrt(np.nansum(all_star_errors**2, axis=0))

sf2_star_fluxes = tot_star_fluxes[1]

# get spectral response

## read in the star model
phoenix_dir = "/home/jwang/scratch/PHOENIX/"
hdulist = fits.open(phoenix_dir + "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
star_model_wvs = hdulist[0].data
hdulist.close()

hdulist = fits.open(phoenix_dir + "lte05600-3.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits", ignore_missing_end=True)
star_model = hdulist[0].data
hdulist.close()

star_radius = hdulist[0].header['PHXREFF'] #cm
star_model_wvs /= 10000

nir = np.where((star_model_wvs >= 0.9) &(star_model_wvs < 5) )
star_model_wvs = star_model_wvs[nir]
star_model = star_model[nir]
star_model_r = np.median(star_model_wvs)/np.median(star_model_wvs - np.roll(star_model_wvs, 1))
star_model_downsample = star_model_r/35000/(2*np.sqrt(2*np.log(2)))
print("downsample", star_model_downsample)

star_model = ndi.gaussian_filter(star_model, star_model_downsample)


response =  kpicdrp.extraction.measure_spectral_response(sf2_wvs, sf2_star_fluxes, star_model_wvs, star_model)

# read in the template
model_dir = "/home/jwang/scratch/bt-settl-cifist2011/"
template_filename = os.path.join(model_dir, "lte025-5.5-0.0a+0.0.BT-Settl.spec.7")

#dat = ascii.read(modeldir + filename)
with open(template_filename, 'r') as f:
    model_wvs = []
    model_fluxes = []
    for line in f.readlines():
        line_args = line.strip().split()
        model_wvs.append(float(line_args[0]))
        model_fluxes.append(float(line_args[1].replace('D', 'E')))
    print(line_args)
dat = table.Table([model_wvs, model_fluxes], names=['col0', 'col1'])

dat['col0'] = dat['col0'] * 1.
dat['col0'] *= u.Angstrom.to(u.um)

nir = np.where( (dat['col0'] > 0.8) & (dat['col0'] < 6))
dat = dat[nir]

model_wvs = dat['col0']

grid_wvs = model_wvs[np.where((model_wvs > 2.2) & (model_wvs < 2.5))]

dat['col1'] = 10**(dat['col1'] - 8)
dat['col1'] *= (u.erg/u.cm**2/u.s/u.Angstrom).to(u.W/u.m**2/u.um)
model_flux = dat['col1']

# xcorr time
orders = np.array([6,7])
shifts = np.linspace(-500, 500, 100)


# ccf, acf = xcorr.simple_xcorr(shifts, sf2_wvs[orders], sf2_fluxes[orders], model_wvs, model_flux)

ccf, acf, _ = xcorr.lsqr_xcorr(shifts, sf2_wvs[orders], sf2_fluxes[orders], sf2_wvs[orders].ravel(), sf2_star_fluxes[orders].ravel(), model_wvs, model_flux, orders_responses=response[orders])
  
plt.figure()
plt.plot(shifts, ccf)
plt.plot(shifts, acf * np.max(ccf)/np.max(acf))
plt.show()

