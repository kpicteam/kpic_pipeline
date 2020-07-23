import numpy as np
import itertools
from copy import copy
from glob import glob
import os
import astropy.io.fits as pyfits

def _task_convolve_spectrum_pixel_width(paras):
    indices,wvs,spectrum,pixel_widths = paras

    conv_spectrum = np.zeros(np.size(indices))
    dwvs = wvs[1::]-wvs[0:(np.size(wvs)-1)]
    dwvs = np.append(dwvs,dwvs[-1])
    for l,k in enumerate(indices):
        pwv = wvs[k]
        pix_width = pixel_widths[k]
        w = int(np.round(pix_width/dwvs[k]*2))
        stamp_spec = spectrum[np.max([0,k-w]):np.min([np.size(spectrum),k+w])]
        stamp_wvs = wvs[np.max([0,k-w]):np.min([np.size(wvs),k+w])]
        stamp_rel_wvs =  np.abs(stamp_wvs-pwv)
        hatkernel = np.array(stamp_rel_wvs<pix_width).astype(np.int)
        conv_spectrum[l] = np.sum(hatkernel*stamp_spec)/np.sum(hatkernel)
    return conv_spectrum

def convolve_spectrum_pixel_width(wvs,spectrum,pixel_widths,mypool=None):
    if mypool is None:
        return _task_convolve_spectrum_pixel_width((np.arange(np.size(spectrum)).astype(np.int),wvs,spectrum,pixel_widths))
    else:
        conv_spectrum = np.zeros(spectrum.shape)

        chunk_size=100
        N_chunks = np.size(spectrum)//chunk_size
        indices_list = []
        for k in range(N_chunks-1):
            indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int))
        indices_list.append(np.arange((N_chunks-1)*chunk_size,np.size(spectrum)).astype(np.int))
        outputs_list = mypool.map(_task_convolve_spectrum_pixel_width, zip(indices_list,
                                                               itertools.repeat(wvs),
                                                               itertools.repeat(spectrum),
                                                               itertools.repeat(pixel_widths)))
        for indices,out in zip(indices_list,outputs_list):
            conv_spectrum[indices] = out

        return conv_spectrum

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
        # if k%100 == 0:
        #     plt.plot(stamp_wvs,gausskernel/np.nanmax(gausskernel))
        #     plt.plot(stamp_wvs,stamp_spec/np.nanmax(stamp_spec))
        #     plt.plot(stamp_wvs,conv_spectrum[np.max([0,k-w]):np.min([np.size(wvs),k+w])]/np.nanmax(conv_spectrum[np.max([0,k-w]):np.min([np.size(wvs),k+w])]))
        #     plt.show()
    return conv_spectrum

def convolve_spectrum_line_width(wvs,spectrum,line_widths,mypool=None):
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


def combine_stellar_spectra(spectra,errors,weights=None):
    if weights is None:
        _weights = np.ones(spectra.shape[0])/float(spectra.shape[0])
    else:
        _weights = weights
    cp_spectra = copy(spectra)*_weights[:,None,None]

    flux_per_spectra = np.nansum(cp_spectra, axis=2)[:,:,None]

    scaling4badpix = (np.nansum(flux_per_spectra,axis=0)/np.sum(np.isfinite(cp_spectra)*flux_per_spectra,axis=0))
    # print(scaling4badpix.shape)
    scaling4badpix[np.where(scaling4badpix>2)] = np.nan
    # exit()
    med_spec = np.nansum(cp_spectra, axis=0)*scaling4badpix
    errors = np.sqrt(np.nansum((errors*_weights[:,None,None])**2, axis=0))*scaling4badpix
    return med_spec,errors

def combine_science_spectra(spectra,errors):
    # deno = np.nansum(1/errors**2, axis=0)
    # out_spec = np.nansum(spectra/errors**2, axis=0)/deno
    # out_errors = 1/np.sqrt(deno)
    out_spec = np.nanmean(spectra, axis=0)
    out_errors = np.sqrt(np.nansum(errors**2, axis=0))/spectra.shape[0]
    # out_errors = np.ones(out_errors.shape)*np.nanmedian(out_errors)
    return out_spec,out_errors

def edges2nans(spec):
    cp_spec = copy(spec)
    cp_spec[:,:,0:5] = np.nan
    cp_spec[:,:,2000::] = np.nan
    return cp_spec

def combine_spectra_from_folder(filelist,mode,science_mjd=None): #"star"


    fluxes_list = []
    errors_list = []
    slits_list = []
    darks_list = []
    # fiber_list = []
    header_list = []
    baryrv_list = []
    fiber_list = []
    mjd_list = []
    for filename in filelist:  #
        # print(filename)
        hdulist = pyfits.open(filename)
        fluxes = hdulist[0].data
        header = hdulist[0].header
        errors = hdulist[1].data
        slits = hdulist[2].data
        darks = hdulist[3].data

        # fiber_list.append(np.argmax(np.nansum(fluxes, axis=(1, 2))))
        baryrv_list.append(float(header["BARYRV"]))
        mjd_list.append(float(header["MJD"]))
        fluxes_list.append(fluxes)
        errors_list.append(errors)
        slits_list.append(slits)
        darks_list.append(darks)
        header_list.append(header)
        fiber_list.append(np.argmax(np.nansum(fluxes, axis=(1, 2))))
    #     import matplotlib.pyplot as plt
    #     for fib in range(3):
    #         plt.figure(1+fib)
    #         for order_id in range(9):
    #             plt.subplot(9, 1, 9-order_id)
    #             plt.plot(fluxes[fib,order_id,:],linestyle="-",linewidth=2,label="data")
    #             print(fib, order_id, np.nanmedian(fluxes[fib,order_id,:]))
    #             # plt.plot(fluxes_slit[4,order_id,:],linestyle="-",linewidth=2,label="slit")
    #             # plt.plot(fluxes_dark[4,order_id,:],linestyle="-",linewidth=2,label="dark")
    #             # plt.plot(errors[0,order_id,:],linestyle="--",linewidth=2)
    #             plt.legend()
    #
    #         # plt.figure(10+fib)
    #         # plt.imshow(residuals,interpolation="nearest",origin="lower")
    #         # plt.clim(-0,100)
    # plt.show()
    # fiber_list = np.array(fiber_list)
    fluxes_list = np.array(fluxes_list)
    errors_list = np.array(errors_list)
    slits_list = np.array(slits_list)
    darks_list = np.array(darks_list)
    baryrv_list = np.array(baryrv_list)
    fiber_list = np.array(fiber_list)
    mjd_list = np.array(mjd_list)

    if science_mjd is not None:
        dt_list = np.abs(science_mjd-mjd_list)
        weights = 1-dt_list/np.max([np.max(dt_list),1./24.])
        weights /= np.nansum(weights)
        # print(weights)
        # exit()
    else:
        weights = np.ones(len(mjd_list))/float(len(mjd_list))

    combined_spec = np.zeros(fluxes.shape)
    combined_spec_sig = np.zeros(fluxes.shape)
    combined_slit = np.zeros(fluxes.shape)
    combined_dark = np.zeros((fluxes.shape[0]*2,fluxes.shape[1],fluxes.shape[2]))
    for fib in range(fluxes.shape[0]):
        if mode == "star":
            where_fib = np.where(fiber_list==fib)[0]
            # print(where_fib)
            # exit()
            if len(where_fib) != 0:
                combined_spec[fib, :, :], combined_spec_sig[fib, :, :] = combine_stellar_spectra(fluxes_list[where_fib, fib, :, :], errors_list[where_fib, fib, :, :],weights=weights[where_fib])
        elif mode == "science":
            combined_spec[fib, :, :], combined_spec_sig[fib, :, :] = combine_science_spectra(fluxes_list[:, fib, :, :], errors_list[:, fib, :, :])
            combined_slit[fib, :, :], _ = combine_science_spectra(slits_list[:, fib, :, :], errors_list[:, fib, :, :])
            combined_dark[fib, :, :], _ = combine_science_spectra(darks_list[:, fib, :, :], errors_list[:, fib, :, :])
            combined_dark[fib+3, :, :], _ = combine_science_spectra(darks_list[:, fib+3, :, :], errors_list[:, fib, :, :])

    # exit()
    combined_spec = edges2nans(combined_spec)
    combined_spec_sig = edges2nans(combined_spec_sig)
    if mode == "star":
        return combined_spec, combined_spec_sig, None,None,np.mean(baryrv_list)
    elif mode == "science":
        combined_slit = edges2nans(combined_slit)
        combined_dark = edges2nans(combined_dark)

        return combined_spec, combined_spec_sig, combined_slit,combined_dark,np.mean(baryrv_list)