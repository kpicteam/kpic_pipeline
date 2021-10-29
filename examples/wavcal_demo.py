import os
import multiprocessing as mp
from glob import glob
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import kpicdrp.wavecal as wavecal
import kpicdrp.utils as utils
import kpicdrp.data as data
from kpicdrp.caldb import trace_caldb, wave_caldb
import pandas as pd
from scipy.interpolate import interp1d



## Change local directory
kpicpublicdir = "fill/in/your/path/public_kpic_data/" # main data dir

## Path relative to the public kpic directory
filelist_spectra = glob(os.path.join(kpicpublicdir, "20200928_HIP_95771","fluxes", "*bkgdsub_spectra.fits"))

filename_oldwvs = os.path.join(kpicpublicdir, "utils", "first_guess_wvs_20200928_HIP_81497.fits")
filename_phoenix_rvstandard = os.path.join(kpicpublicdir, "utils", "HIP_81497_lte03600-1.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
filename_phoenix_wvs =os.path.join(kpicpublicdir, "utils", "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")

use_atran = False # Use atran model for tellurics
filelist_atran = glob(os.path.join(kpicpublicdir, "utils","atran","atran_13599_30_*.dat"))
use_psg = True # Use psg model for tellurics
psg_filename = os.path.join(kpicpublicdir,"utils","psg",'psg_out_2020.12.18_l0_1900nm_l1_2600nm_lon_204.53_lat_19.82_pres_0.5826.fits')

# define output directory
if use_psg:
    out_filename = os.path.join(kpicpublicdir, "20200928_HIP_95771", "calib", "20200928_HIP_95771_psg_wvs.fits")
if use_atran:
    out_filename = os.path.join(kpicpublicdir, "20200928_HIP_95771", "calib", "20200928_HIP_95771_atran_wvs.fits")
if not os.path.exists(os.path.join(kpicpublicdir, "20200928_HIP_95771", "calib")):
    os.makedirs(os.path.join(kpicpublicdir, "20200928_HIP_95771", "calib"))

# wavelength solution parameters
target_rv = -85.391 # HIP_95771: -85.391km/s ; HIP_81497: -55.567 #km/s
N_nodes_wvs=6 # Number of spline nodes for the wavelength solution
blaze_chunks=5 # Number of chunks of the "blaze profile" (modeled as a piecewise linear function)
init_grid_search = True # do a rough grid search of the wavcal before running optimizer
init_grid_dwv = 1e-4#3e-4 #microns, how far to go for the grid search. Caution: It can take quite a while!
fringing = False
numthreads = 2
mypool = mp.Pool(processes=numthreads)
# mypool = None

# Read an old wavelength array to be used as the first guess
# hdulist = fits.open(filename_oldwvs)
# old_wvs = hdulist[0].data
old_wvsoln = data.Wavecal(filepath=filename_oldwvs)

spectral_dataset = data.Dataset(filelist=filelist_spectra, dtype=data.Spectrum)

trace_dat = trace_caldb.get_calib(spectral_dataset[0])

# combine spectral dataset together to a single specrum per fiber
combined_spectrum = utils.stellar_spectra_from_files(spectral_dataset)

# uncomment to only reduce a single order for test purposes
# combined_spec,combined_err,line_width,old_wvs = combined_spec[:,0:1,:],combined_err[:,0:1,:],line_width[:,0:1,:],old_wvs[:,0:1,:]
# combined_spec,combined_err,line_width,old_wvs = combined_spec[:,6:7,:],combined_err[:,6:7,:],line_width[:,6:7,:],old_wvs[:,6:7,:]
# combined_spec,combined_err,line_width,old_wvs = combined_spec[:,8:9,:],combined_err[:,8:9,:],line_width[:,8:9,:],old_wvs[:,8:9,:]

# print(combined_spec.shape)
# utils.plot_kpic_spectrum(combined_spec[1,:,:])
# plt.show()

# Read the Phoenix model and wavelength corresponding to the RV standard star.
with fits.open(filename_phoenix_wvs) as hdulist:
    phoenix_wvs = hdulist[0].data / 1.e4
crop_phoenix = np.where((phoenix_wvs > 1.8 - (2.6 - 1.8) / 2) * (phoenix_wvs < 2.6 + (2.6 - 1.8) / 2))
phoenix_wvs = phoenix_wvs[crop_phoenix]
with fits.open(filename_phoenix_rvstandard) as hdulist:
    phoenix_spec = hdulist[0].data[crop_phoenix]

new_wvs_arr = np.zeros(combined_spectrum.data.shape)
line_width_func_list = utils.linewidth2func(trace_dat.widths[trace_dat.get_sci_indices()], old_wvsoln.wvs)

fib_labels = []
for fib in range(combined_spectrum.fluxes.shape[0]):
    if np.nansum(combined_spectrum.fluxes[fib,:,:])==0:
        continue
    baryrv_fib = combined_spectrum.header['BARYRV{0}'.format(fib)]
    star_rv = target_rv - baryrv_fib

    # broaden and create interpolation function for the Phoenix model
    phoenix_line_widths = np.array(pd.DataFrame(line_width_func_list[fib](phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
    print("broaden Phoenix model")
    phoenix_conv = utils.convolve_spectrum_line_width(phoenix_wvs, phoenix_spec, phoenix_line_widths, mypool=mypool)
    ##
    phoenix_func = interp1d(phoenix_wvs, phoenix_conv / np.nanmax(phoenix_conv), bounds_error=False,fill_value=np.nan)

    if use_atran: #atran
        ## Create a interpolation function for the tellurics model (including water and zenith angle)
        atrangridname = os.path.join(kpicpublicdir, "utils","atran", "atran_grid_f{0}.fits".format(fib))
        if 0:
            # enable this section to broaden and save the atran grid of models.
            # The resulting fits file can directly be used to generate a regular grid interpolator as shown below.
            # the broadening is specific to the line width calibration, so it might vary from epoch to epoch and maybe
            # even between fibers. That being said, it might not make a big difference...
            wavecal.save_atrangrid(filelist_atran, line_width_func_list[fib], atrangridname,mypool=mypool)
            # exit()
        hdulist = fits.open(atrangridname)
        atran_grid =  hdulist[0].data
        water_unique =  hdulist[1].data
        angle_unique =  hdulist[2].data
        atran_wvs =  hdulist[3].data
        hdulist.close()
        ##
        print("atran grid interpolator")
        atran_interpgrid = RegularGridInterpolator((water_unique,angle_unique),atran_grid,method="linear",bounds_error=False,fill_value=0.0)

        # Derive wavecal for a single fiber
        print("start fitting")
        new_wvs_fib,model,out_paras = wavecal.fit_wavecal_fib(old_wvsoln.wvs[fib,:,:], combined_spectrum.fluxes[fib,:,:], combined_spectrum.errs[fib,:,:],
                                                        phoenix_func,star_rv,atran_wvs,atran_interpgrid,
                                                        N_nodes_wvs=N_nodes_wvs,
                                                        blaze_chunks=blaze_chunks,
                                                        init_grid_search = init_grid_search,
                                                        init_grid_dwv = init_grid_dwv,
                                                        fringing=fringing,
                                                        mypool=mypool)
    if use_psg:
        # define things for example
        l0, l1   = 1900,2600 # bounds to compute the telluric model in nm

        wvs_psg, psg_tuple = wavecal.open_psg_allmol(psg_filename,l0,l1) # return x array and psg spectra returned in a tuple
        wvs_psg /= 1000 # convert from nm to um

        # Derive wavecal for a single fiber
        print("start fitting")
        new_wvs_fib,model,out_paras = wavecal.fit_psg_wavecal_fib(old_wvsoln.wvs[fib,:,:], combined_spectrum.fluxes[fib,:,:], combined_spectrum.errs[fib,:,:],
                                                        phoenix_func,star_rv,wvs_psg,psg_tuple,
                                                        N_nodes_wvs=N_nodes_wvs,
                                                        blaze_chunks=blaze_chunks,
                                                        init_grid_search = init_grid_search,
                                                        init_grid_dwv = init_grid_dwv,
                                                        fringing=fringing,
                                                        mypool=mypool)
        print(out_paras)

    new_wvs_arr[fib,:,:] = new_wvs_fib
    fib_labels.append(combined_spectrum.labels[fib])

    plt.figure(fib+1,figsize=(12,12))
    ax_list = utils.plot_kpic_spectrum(combined_spectrum.fluxes[fib,:,:] ,wvs=new_wvs_fib, arr_err=combined_spectrum.errs[fib,:,:],color="blue",label="data")
    ax_list = utils.plot_kpic_spectrum(model,wvs=new_wvs_fib,color="orange",label="model",ax_list=ax_list)
    plt.legend()
    print("Saving " + out_filename.replace(".fits","_f{0}.png".format(fib)))
    plt.savefig(out_filename.replace(".fits","_f{0}.png".format(fib)))
    # plt.show()


new_wvs = data.Wavecal(wvs=new_wvs_arr, header=combined_spectrum.header, labels=fib_labels, method="Star", filepath=out_filename)
new_wvs.save(caldb=wave_caldb)

plt.show()
