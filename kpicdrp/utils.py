import numpy as np
from copy import copy
import astropy.io.fits as fits
from scipy.interpolate import interp1d
import itertools

def combine_stellar_spectra(spectra,errors,weights=None):
    """
    Combine stellar spectra accounting for total flux variations and nans.

    Args:
        spectra: (Nspec, Norders,Npix) numpy array containing the spectra
        errors: (Nspec, Norders,Npix) numpy array containing the uncertainty
        weights: user defined weigths for each spectrum. Default is constant.

    Returns:
        combined_spec: (Norders,Npix) numpy array for combined spectrum
        combined_errors:  (Norders,Npix) numpy array for combined errors
    """

    if weights is None:
        _weights = np.ones(spectra.shape[0])/float(spectra.shape[0])
    else:
        _weights = weights
    cp_spectra = copy(spectra)*_weights[:,None,None]

    flux_per_spectra = np.nansum(cp_spectra, axis=2)[:,:,None]

    deno = np.sum(np.isfinite(cp_spectra)*flux_per_spectra,axis=0)
    where_null = np.where(deno==0)
    deno[where_null] = np.nan
    scaling4badpix = np.nansum(flux_per_spectra,axis=0)/deno
    scaling4badpix[where_null] = np.infty
    scaling4badpix[np.where(scaling4badpix>2)] = np.nan
    combined_spec = np.nansum(cp_spectra, axis=0)*scaling4badpix
    combined_errors = np.sqrt(np.nansum((errors*_weights[:,None,None])**2, axis=0))*scaling4badpix

    return combined_spec,combined_errors

def linewidth2func(line_width,wvs):
    """"
    Transforms line width array into a list of interp1d functions wrt wavelength
    Args:
        line_width: (Nfibers Norders,Npix)  linewidth array (sigma of a 1D gaussian)
        wvs: (Nfibers Norders,Npix) wavelength array

    Returns:
        line_width_func_list: list of interp1d functions.
            E.g.: line_width_vec = line_width_func_list[1](wv_vec)
    """
    line_width_func_list = []
    for fib in range(line_width.shape[0]):
        dwvs = wvs[fib][:, 1:2048] - wvs[fib][:, 0:2047]
        dwvs = np.concatenate([dwvs, dwvs[:, -1][:, None]], axis=1)
        line_width_wvunit = line_width[fib, :, :] * dwvs
        line_width_func = interp1d(np.ravel(wvs[fib]), np.ravel(line_width_wvunit), bounds_error=False,fill_value=np.nan)
        line_width_func_list.append(line_width_func)
    return line_width_func_list


def _task_convolve_spectrum_line_width(paras):
    indices,wvs,spectrum,line_widths = paras

    conv_spectrum = np.zeros(np.size(indices))
    dwvs = wvs[1::]-wvs[0:(np.size(wvs)-1)]
    dwvs = np.append(dwvs,dwvs[-1])
    for l,k in enumerate(indices):
        pwv = wvs[k]
        sig = line_widths[k]
        w = int(np.round(sig/dwvs[k]*10.))
        stamp_spec = spectrum[np.max([0,k-w]):np.min([np.size(spectrum),k+w])]
        stamp_wvs = wvs[np.max([0,k-w]):np.min([np.size(wvs),k+w])]
        stamp_dwvs = stamp_wvs[1::]-stamp_wvs[0:(np.size(stamp_spec)-1)]
        gausskernel = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(stamp_wvs-pwv)**2/sig**2)
        conv_spectrum[l] = np.sum(gausskernel[1::]*stamp_spec[1::]*stamp_dwvs)
    return conv_spectrum

def convolve_spectrum_line_width(wvs,spectrum,line_widths,mypool=None):
    """
    Convolve spectrum with pixel dependent line width.

    Args:
        wvs: vector of wavelengths
        spectrum: vector spectrum
        line_width: vector of spectral line width (sigma of a 1D gaussian)
        mypool:

    Returns:
        conv_spectrum: broadened spectrum
    """
    if mypool is None:
        return _task_convolve_spectrum_line_width((np.arange(np.size(spectrum)).astype(np.int),wvs,spectrum,line_widths))
    else:
        conv_spectrum = np.zeros(spectrum.shape)

        chunk_size=100
        N_chunks = np.size(spectrum)//chunk_size
        indices_list = []
        for k in range(N_chunks-1):
            indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int))
        indices_list.append(np.arange((N_chunks-1)*chunk_size,np.size(spectrum)).astype(np.int))
        outputs_list = mypool.map(_task_convolve_spectrum_line_width, zip(indices_list,
                                                               itertools.repeat(wvs),
                                                               itertools.repeat(spectrum),
                                                               itertools.repeat(line_widths)))
        for indices,out in zip(indices_list,outputs_list):
            conv_spectrum[indices] = out

        return conv_spectrum


def stellar_spectra_from_files(filelist):
    """"
    Combines star observations from a list of files.
    For a given fiber, it identifies the files where the star was observed on this particular fiber, and combines the
    corresponding spectra.
    The final output has the combined spectrum for each fiber.

    Args:
        filelist: list of filenames
    Returns:
        combined_spec: (Nfibers Norders,Npix) Combined spectra
        combined_err: (Nfibers Norders,Npix) Combined error
    """
    spec_list = []
    err_list = []
    baryrv_list = []
    for filename in filelist:
        hdulist = fits.open(filename)
        spec_list.append(hdulist[0].data[None,:,:,:])
        err_list.append(hdulist[1].data[None,:,:,:])
        baryrv_list.append(float(hdulist[0].header["BARYRV"]))
    all_spec_arr = np.concatenate(spec_list)
    all_err_arr = np.concatenate(err_list)
    baryrv_list = np.array(baryrv_list)

    whichfiber = np.argmax(np.nansum(all_spec_arr, axis=(2, 3)),axis=1)

    combined_spec = np.zeros(all_spec_arr.shape[1::])
    combined_err = np.zeros(all_spec_arr.shape[1::])
    baryrv = np.zeros(all_spec_arr.shape[1])
    for fib in np.unique(whichfiber):
        spec_list = all_spec_arr[np.where(fib == whichfiber)[0], fib, :, :]
        spec_sig_list = all_err_arr[np.where(fib == whichfiber)[0], fib, :, :]
        baryrv[fib] = np.mean(baryrv_list[np.where(fib == whichfiber)[0]])
        combined_spec[fib, :, :], combined_err[fib, :, :] = combine_stellar_spectra(spec_list, spec_sig_list)
    return combined_spec,combined_err,baryrv
