import os
import numpy as np
import astropy.io.ascii
import astropy.modeling.models as models
import astropy.units as u
import kpicdrp

gain = kpicdrp.kpic_params.getfloat('NIRSPEC', 'gain')

k_filt = astropy.io.ascii.read(os.path.join(kpicdrp.datadir, "2massK.txt"), names=['wv', 'trans'])


def calculate_peak_throughput(spectrum, k_mag, bb_temp=5000, fib=None):
    """
    Roughly estimatels throughput of data. Currently only works for K-band for one particular grating configuration!!!

    Args:
        spectrum (data.Spectrum): a wavelength-calibrated 1D spectrum (run spectrum.calibrate_wvs() first!)
        k_mag (float): 2MASS K-band magnitude of the source
        bb_temp (float): optional, blackbody temperature assumed for stellar model. Assume 5000 K otherwise
        fib (str): optional, label for which fiber to evaluate the throughput for. 
                   If not specified, picks the one with highest flux

    Returns
        throughout (float): the 95% highest throughput calculated. Nearly the peak throughput
    """
    # figure out the fiber
    if fib is None:
        med_flux = np.nanmedian(spectrum.data, axis=(1,2))
        fib = spectrum.labels[np.argmax(med_flux)]

    exptime = spectrum.header['TRUITIME'] # exptime in seconds

    bb = models.BlackBody(temperature=bb_temp*u.K)
    star_model = bb(spectrum.wvs * u.um).to(u.W/u.cm**2/u.um/u.sr, equivalencies=u.spectral_density(spectrum.wvs * u.um)).value

    star_model_filt = bb(k_filt['wv'] * u.um).to(u.W/u.cm**2/u.um/u.sr, equivalencies=u.spectral_density(k_filt['wv'] * u.um)).value

    k_zpt = 4.283E-14 # W / cm^2 / micron

    k_flux = k_zpt * 10**(-k_mag/2.5)

    integral = np.sum(star_model_filt * k_filt['trans'])/np.sum(k_filt['trans'])

    norm = k_flux/integral

    photon_energy = 6.626068e-34 * 299792458 / (spectrum.wvs * 1e-6) # Joules
    tele_size = 76 * (100)**2 # cm^2
    model_photonrate = star_model * norm / photon_energy * (tele_size)


    throughputs = []

    # plt.figure()
    for wvs, order in zip(spectrum.wvs[spectrum.trace_index[fib]], spectrum.data[spectrum.trace_index[fib]]):
        xcoords = np.arange(order.shape[0])
        
        dlam = wvs - np.roll(wvs, 1)
        dlam[0] = wvs[1] - wvs[0]
        
        model_photonrate_order = np.interp(wvs, spectrum.wvs.ravel(), model_photonrate.ravel())
        model_photonrate_order *= exptime * dlam
        
        data_photons = order * gain
        
        throughputs.append(data_photons/model_photonrate_order)

    throughputs = np.array(throughputs)

    return np.nanpercentile(throughputs, 95)
