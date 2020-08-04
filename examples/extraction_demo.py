
import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.table as table
import scipy.ndimage as ndi
import kpicdrp.extraction 


with fits.open("/home/jwang/Documents/KPIC/sv-2020-07/200701_bkgd_1.5s.fits") as hdulist:
    bkgd_1s, bkgd_noise_1s, badpixmap_1s = np.copy(hdulist[0].data)



with fits.open("/home/jwang/Documents/KPIC/sv-2020-07/calib/ups_Her_200702_fiber2_trace.fits") as hdulist:
    sf2_trace_centers = hdulist[0].data
    sf2_trace_centers = sf2_trace_centers.reshape([1, sf2_trace_centers.shape[0], sf2_trace_centers.shape[1]])
    sf2_trace_sigmas = hdulist[1].data
    sf2_trace_sigmas = sf2_trace_sigmas.reshape(sf2_trace_centers.shape)

###### Fiber 2 ###########
filestr = "/home/public/nas/kpic/200702/spec/nspec200702_0{0:03d}.fits"
filenums =  list(range(2, 6+1))
filelist = [filestr.format(i) for i in filenums]

nobadpix = np.zeros(bkgd_1s.shape)
sci_frame, sci_noise, sci_frames = kpicdrp.extraction.process_sci_raw2d(filelist, bkgd_1s, nobadpix, detect_cosmics=False)


fluxes, error = kpicdrp.extraction.extract_flux(np.rot90(sci_frame, -1), "extraction_demo.fits", sf2_trace_centers, sf2_trace_sigmas, fit_background=True, bad_pixel_fraction=0.03)

sf2_fluxes = fluxes[0]

import matplotlib.pylab as plt
fig = plt.figure(figsize=(10, 10))
i = 0
for flux_order in sf2_fluxes:
    i += 1
    fig.add_subplot(9, 1, i)
    plt.plot(flux_order, 'b-')
    plt.ylim([0, np.nanpercentile(flux_order, 98) * 1.05])

plt.show()