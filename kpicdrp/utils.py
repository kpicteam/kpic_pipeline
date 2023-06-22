import os
from glob import glob
from copy import copy
import itertools
import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import astropy.time as time
import astropy.io.fits as pyfits
import kpicdrp.data as data

def get_spline_model(x_knots,x_samples,spline_degree=3):
    M = np.zeros((np.size(x_samples),(np.size(x_knots))))
    for chunk in range(np.size(x_knots)):
        tmp_y_vec = np.zeros(np.size(x_knots))
        tmp_y_vec[chunk] = 1
        spl = InterpolatedUnivariateSpline(x_knots, tmp_y_vec, k=spline_degree, ext=0)
        M[:,chunk] = spl(x_samples)
    return M

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
        return _task_convolve_spectrum_line_width((np.arange(np.size(spectrum)).astype(np.int_),wvs,spectrum,line_widths))
    else:
        conv_spectrum = np.zeros(spectrum.shape)

        chunk_size=100
        N_chunks = np.size(spectrum)//chunk_size
        indices_list = []
        for k in range(N_chunks-1):
            indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int_))
        indices_list.append(np.arange((N_chunks-1)*chunk_size,np.size(spectrum)).astype(np.int_))
        outputs_list = mypool.map(_task_convolve_spectrum_line_width, zip(indices_list,
                                                               itertools.repeat(wvs),
                                                               itertools.repeat(spectrum),
                                                               itertools.repeat(line_widths)))
        for indices,out in zip(indices_list,outputs_list):
            conv_spectrum[indices] = out

        return conv_spectrum


def stellar_spectra_from_files(dataset, use_header=False):
    """"
    Combines extracted stellar spectra from a sequence of observations of the star. 
    For a given fiber, it identifies the files where the star was observed on this particular fiber, and combines the
    corresponding spectra.
    The final output has the combined spectrum for each fiber.

    Args:
        dataset (data.Dataset): a dataset of Spectrum data of the star on various fibers.
        use_header (bool): if True, uses the header keywords to determine which file corresponds to which fiber 
    Returns:
        combined_spec: (Nfibers, Norders, Npix) Combined spectra [note that Nfibers only corresponds to fibers with the star]
        combined_err: (Nfibers, Norders, Npix) Combined error
    """
    spec_list = []
    err_list = []
    for frame in dataset:
        spec_list.append(frame.fluxes[None,:,:,:])
        err_list.append(frame.errs[None,:,:,:])
    all_spec_arr = np.concatenate(spec_list)
    all_err_arr = np.concatenate(err_list)
    baryrv_list = np.array(dataset.get_header_values('BARYRV'))

    if not use_header:
        whichfiber = np.argmax(np.nansum(all_spec_arr, axis=(2, 3)),axis=1)
        unique_fibers = np.sort(np.unique(whichfiber)) # all fibers with starlight
        unique_fiber_labels = [dataset[0].labels[fib] for fib in unique_fibers]
    else:
        unique_fiber_labels = list(dataset.fib_indices.keys())
        whichfiber = np.zeros(len(dataset), dtype=int)
        unique_fibers = []
        for fib, fib_label in enumerate(unique_fiber_labels):
            for index in dataset.fib_indices[fib_label]:
                whichfiber[index] = fib
            unique_fibers.append(fib)
        unique_fibers = np.array(unique_fibers)

    combined_spec = np.zeros(all_spec_arr.shape[1::])
    combined_err = np.zeros(all_spec_arr.shape[1::])
    baryrv = np.zeros(all_spec_arr.shape[1])
    for fib in unique_fibers:
        spec_list = all_spec_arr[np.where(fib == whichfiber)[0], fib, :, :]
        spec_sig_list = all_err_arr[np.where(fib == whichfiber)[0], fib, :, :]
        baryrv[fib] = np.mean(baryrv_list[np.where(fib == whichfiber)[0]])
        combined_spec[fib, :, :], combined_err[fib, :, :] = combine_stellar_spectra(spec_list, spec_sig_list)

    combined_spectra = data.Spectrum(fluxes=combined_spec[unique_fibers], errs=combined_err[unique_fibers], header=dataset[0].header, labels=unique_fiber_labels)
    combined_spectra.filedir = dataset[0].filedir
    combined_spectra.filename = dataset[0].filename.replace(".fits", "_combined.fits")
    tnow = time.Time.now()
    combined_spectra.header.add_history("[{0}] Combined {1} Spectra Together".format(str(tnow), len(dataset)))
    combined_spectra.header['DRPNFILE'] = len(dataset)
    for i, frame in enumerate(dataset):
        combined_spectra.header['FILE{0}'.format(i)] = frame.filename
    combined_spectra.header['BARYRV'] = np.mean(baryrv_list)
    for i, fib in enumerate(np.unique(whichfiber)):
        combined_spectra.header['BARYRV{0}'.format(i)] = baryrv[fib]

    return combined_spectra

def get_avg_mjd_radec(filelist):
    """
    Returns the average MJD of the files, and the RA/Dec the telescope was pointing to

    Args:
        filelist: list of files
    """
    mjds = []
    ras = []
    decs = []

    for filename in filelist:
        with fits.open(filename) as hdulist:
            utctime = hdulist[0].header['UTC']
            date = hdulist[0].header['DATE-OBS']
            ra = hdulist[0].header['RA']
            dec = hdulist[0].header['Dec']

            mjd = time.Time("{0}T{1}Z".format(date, utctime)).mjd

            tel_coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))

            mjds.append(mjd)
            ras.append(tel_coord.ra.degree)
            decs.append(tel_coord.dec.degree)

    return np.mean(mjds), np.mean(ras), np.mean(decs)


keck = EarthLocation.from_geodetic(lat=19.8283*u.deg, lon=-155.4783*u.deg, height=4160*u.m)
def compute_rel_vel(mjd, ra, dec, star_v):
    """
    Computes the relative velocity of the star relative to Earth during the time of observation
    Args:
        mjd: (float) time MJD
        ra: RA in degrees
        dec: Dec in degrees
        star_v: star's radial velocity (km/s)
    Return:
        rel_v: (float) in km/s
    """
    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, )
    barycorr = sc.radial_velocity_correction(obstime=time.Time(mjd, format='mjd', scale='utc'), location=keck)
    print(barycorr.to(u.km/u.s).value )

    rel_v = -barycorr.to(u.km/u.s).value + star_v # postiive is redshfit? 
    
    return rel_v


def plot_kpic_spectrum(arr,wvs=None,arr_err=None,ax_list=None,linestyle="-",linewidth=2,color=None,label=None):
    import matplotlib.pyplot as plt
    if ax_list is None:
        _ax_list = []

    if wvs is None:
        wvs = np.tile(np.arange(arr.shape[1])[None,:],(arr.shape[0],1))

    for order_id in range(arr.shape[0]):
        if ax_list is None:
            plt.subplot(arr.shape[0], 1, arr.shape[0]-order_id)
            _ax_list.append(plt.gca())
        else:
            plt.sca(ax_list[order_id])
        plt.plot(wvs[order_id,:],arr[order_id,:],linestyle=linestyle,linewidth=linewidth,label=label,color=color)
        if arr_err is not None:
            plt.fill_between(wvs[order_id,:],
                             arr[order_id,:] - arr_err[order_id,:],
                             arr[order_id,:] + arr_err[order_id,:],
                             label=label+" (err)", alpha=0.5,color=color)

    if ax_list is None:
        return _ax_list
    else:
        return ax_list

def get_calib_bkg(filename,mybkgdir):
    hdulist = pyfits.open(filename)
    # dat = np.copy(hdulist[0].data)
    # dat = np.rot90(hdulist[0].data, -1)
    header = hdulist[0].header

    tint = float(header["TRUITIME"])
    coadds = int(header["COADDS"])

    # read the master Background file
    background_med_filename = glob(os.path.join(mybkgdir, "*background_med_nobars_tint{0}_coadds{1}.fits".format(tint, coadds)))[0]
    print(background_med_filename)
    _hdulist = pyfits.open(background_med_filename)
    bkgd = _hdulist[0].data
    bkgd_noise = _hdulist[1].data

    # read the bad pixel map
    persisbadpixmap_filename = glob(os.path.join(mybkgdir, "*persistent_badpix_nobars_tint{0}_coadds{1}.fits".format(tint, coadds)))[0]
    print(persisbadpixmap_filename)
    _hdulist = pyfits.open(persisbadpixmap_filename)
    badpixmap = _hdulist[0].data

    return bkgd,bkgd_noise,badpixmap