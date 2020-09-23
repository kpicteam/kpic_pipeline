import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interpolate
import scipy.optimize 
import astropy.io.fits as fits
from astropy.modeling import models, fitting
import astroscrappy
import  astropy.time as time
import astropy.units as u
from astropy.coordinates import SkyCoord

gain = 2.85 # e-/ADU


def process_sci_raw2d(filelist, bkgd, badpixmap, detect_cosmics=True):
    """
    Does simple processing to the raw spectroscopic data: bkgd subtraction, bad pixels, cosmic rays

    Args:
        filelist: list of science files to process
        bkgd: 2-D bkgd frame for subtraction
        badpixmap: 2-D bad pixel map
    Output:
        sci_data: stacked science frame
        sci_noise: noise from std
    """

    # list containing 1 or more frames
    sci_data = []
    # iterate over images from the same fiber
    for filename in filelist:
        with fits.open(filename) as hdulist:
            dat = np.copy(hdulist[0].data)
            dat = np.rot90(dat, -1)

            if detect_cosmics:
                dat_crmap, corr_dat = astroscrappy.detect_cosmics(dat, inmask=badpixmap)
                dat[dat_crmap] = np.nan
            dat[np.where(np.isnan(badpixmap))] = np.nan

            # subtract background
            dat -= bkgd

            sci_data.append(dat)
        hdulist.close()

    # take mean and std across different frames - JX: is the mean used later on?
    sci_noise = np.nanstd(sci_data, axis=0)
    mean_sci_data = np.nanmean(sci_data, axis=0)

    return mean_sci_data, sci_noise, sci_data


def extract_1d(dat_coords, dat_slice, center, sigma, noise):
    """
    Optimal (I think) extraction of a single 1-D slice

    Args:
        dat_coords: y-coordinates of dat_slice (Ny)
        dat_slice: data along a slice (Ny)
        center: center of Gaussian in dat_coords coordinates (float)
        sigma: standard deviation of Gaussian (float)
        noise: noise in each pixel of data (Ny)

    Returns:
        flux: total integrated flux assuming a Gaussian PSF (float)
        badpixmetric metric for how good of a flux estimate this is (float)
    """
    g = models.Gaussian1D(amplitude=1, mean=center, stddev=sigma)
    good = np.where(~np.isnan(dat_slice))
    if np.size(good) < 3:
        return np.nan, np.nan
    good_slice = dat_slice[good]
    good_coords = dat_coords[good]
    noise = noise[good]
    initial_flux = np.sum(good_slice * g(good_coords))/np.sum(g(good_coords)*g(good_coords))
    
    good_var = (initial_flux * g(good_coords))/gain + noise**2
    flux = np.sum(good_slice * g(good_coords) / good_var)/np.sum(g(good_coords)*g(good_coords)/good_var)
    residuals = flux * g(good_coords) - good_slice
    max_res = np.nanmax(np.abs(residuals))
    res_mad = np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))
    badpixmetric = np.abs(max_res/res_mad)

    flux *= np.sqrt(2*np.pi) * sigma # total flux, not just peak flux
    
    # if flux > 200:
    #     import matplotlib.pylab as plt
    #     plt.plot(good_coords, good_slice, 'o')
    #     plt.show()
    #     import pdb; pdb.set_trace()
    #if np.isnan(flux):
    #    import pdb; pdb.set_trace()
    
    return flux, max_res


def _extract_flux_chunk(image, order_locs, order_widths, img_noise, fit_background):
    """
    Extracts the flux from a chunk of size Nx of an order for some given fibers

    Args:
        image: 2-D frame (Ny, Nx)
        order_locs: y-location of fiber traces in an order (N_fibers, Nx)
        order_widths: standard deviation of fiber traces in an order (N_fibers, Nx)
        img_noise; 2-D frame for noise in each pixel (Ny, Nx)
        fit_background: if True, fit the background

    Returns:
        fluxes: (N_fibers, Nx) extracted 1-D fluxes
        errors: (N_fibers, Nx) errors for 1-D extracted fluxes
        badpixmetric: (N_fibers, Nx) metric for how bad a pixel is 
    """
    fluxes = np.zeros(order_locs.shape)+np.nan
    fluxerrs = np.zeros(order_locs.shape)+np.nan
    badpixmetrics = np.zeros(order_locs.shape)+np.nan
    
    nx = order_locs.shape[1]
    num_fibers = order_locs.shape[0]

    # iterate over chunk (size of 8 pixels)
    # compute the background from where there are no fibers
    # we'll use this for estimating the background, and estimate the noise in the flux extraction
    for x in range(nx):
        # a single vertical column (y direction)
        img_column = image[:, x]
        centers = order_locs[:, x]
        bkgd_column = np.copy(img_column) # make a copy for masking the fibers
        # for each fiber, mask 11 pixels around it
        for center in centers:
            center_int = int(np.round(center))
            bkgd_column[center_int-5:center_int+5+1] = np.nan 
        ymin_long = int(np.round(np.min(order_locs[:, x]))) - 6
        ymax_long = int(np.round(np.max(order_locs[:, x]))) + 6 + 1
        bkgd_slice = bkgd_column[ymin_long:ymax_long]
        bkgd_noise = np.nanstd(bkgd_slice) # roughly the noise

        if fit_background:
            # compute background as median of remaining pixels
            bkgd_level = np.nanmedian(bkgd_slice)
        else:
            bkgd_level = 0

        column_maxres = []
        for fiber in range(num_fibers):
            center = order_locs[fiber, x]
            sigma = order_widths[fiber, x]
            center_int = int(np.round(center))

            # take a slice of data and subtract the background
            # JX: Why 6?
            dat_slice = img_column[center_int-6:center_int+6+1] - bkgd_level
            ys = np.arange(center_int-6, center_int+6+1)
            noise = img_noise[center_int-6:center_int+6+1, x]

            # JX: when would this be nan? on bad pixels?
            if np.any(np.isnan(img_column[center_int-1:center_int+2])):
                flux, maxres = np.nan, np.nan
            else:
                #flux, badpixmetric = extract_1d(ys, dat_slice, center, sigma, noise)
                flux, maxres = extract_1d(ys, dat_slice, center, sigma, noise)

            column_maxres.append(maxres)
            fluxes[fiber, x] = flux
            # account for the fact we are doing total integrated flux of Gaussian when computing the noise
            # JX: sometimes flux is too negative and makes arg of sqrt negative, throwing Runtime warning
            fluxerr = np.sqrt(gain*flux + (bkgd_noise * np.sqrt(2*np.pi) * sigma)**2 )
            fluxerrs[fiber, x] = fluxerr
        column_maxres = np.array(column_maxres)    
        
        ymin_long = int(np.round(np.min(order_locs[:, x]))) - 6
        ymax_long = int(np.round(np.max(order_locs[:, x]))) + 6 + 1
        ys_long = np.arange(ymin_long, ymax_long)
        long_slice = img_column[ymin_long:ymax_long] - bkgd_level
        for fiber in range(num_fibers):
            sigma = order_widths[fiber, x]
            peakflux = fluxes[fiber, x]/(np.sqrt(2*np.pi) * sigma)
            g = models.Gaussian1D(amplitude=peakflux, mean=order_locs[fiber, x], stddev=sigma)
            long_slice -= g(ys_long)
        
        these_badmetrics = column_maxres/np.nanstd(long_slice)
        badpixmetrics[:, x] = these_badmetrics

    return fluxes, fluxerrs, badpixmetrics
        

def extract_flux(image, output_filename, trace_locs, trace_widths,
                 img_noise=None, img_hdr=None, fit_background=False, bad_pixel_fraction=0.0, pool=None):
    """
    Extracts the flux from the traces to make 1-D spectra. 

    Args:
        image: either 1) 2-D frame (Ny, Nx), or 2) filename of FITS file to 2-D frame
        output_filename: path to output directory
        trace_locs: y-location of the fiber traces (N_fibers, N_orders, Nx)
        trace_widths: standard deviation widths of the fiber traces (N_fibers, N_orders, Nx)
        img_noise: optional frame that describes the noise in each pixel (Ny, Nx). Either 2-D array, or filename of FITS file
        img_hdr: optional FITS header for the image
        fit_background: if True, fits for a constant background during trace extraction (default: False). 
        bad_pixel_fraction: assume this fraction of all 1-D fluxes are bad and will be masked as nans (default 0.01)
        pool: optional multiprocessing.Pool object to pass in to parallelize the spectral extraction (default: None)

    Return:
        Saved also in output_filename as a fits file:
        fluxes: (N_fibers, N_orders, Nx) extracted 1-D fluxes. 
        errors: (N_fibers, N_orders, Nx) errors on extracted 1-D fluxes.
    """
    if not isinstance(image, np.ndarray):
        with fits.open(image) as hdulist:
            image = hdulist[0].data
            img_hdr = hdulist[0].header

    if img_noise is None:
        img_noise = np.ones(image.shape)
    else:
        if not isinstance(img_noise, np.ndarray):
            with fits.open(img_noise) as hdulist:
                img_noise = hdulist[0].data

    # output
    fluxes = np.zeros(trace_locs.shape)+np.nan
    errors = np.zeros(trace_locs.shape)+np.nan
    badpixmetric = np.zeros(trace_locs.shape)+np.nan

    num_fibers = trace_locs.shape[0]
    num_orders = trace_locs.shape[1]
    nx = trace_locs.shape[2]
    # break up image into chunks
    chunk_size = 8
    num_chunks = nx // 8

    pool_jobs = [] # for multiprocessing.Pool if needed

    # iterate over each order
    for order in range(num_orders):
        order_locs = trace_locs[:, order]
        order_widths = trace_widths[:, order]
        # go over each chunk within an order
        for chunk in range(num_chunks):
            c_start = chunk * chunk_size
            c_end = (chunk + 1) * chunk_size
            img_chunk = image[:, c_start:c_end]
            chunk_locs = order_locs[:, c_start:c_end]
            chunk_widths = order_widths[:, c_start:c_end]
            noise_chunk = img_noise[:, c_start:c_end]

            if pool is None:
                # extract flux from chunk
                outputs = _extract_flux_chunk(img_chunk, chunk_locs, chunk_widths, noise_chunk, fit_background)
                fluxes[:, order, c_start:c_end] = outputs[0]
                errors[:, order, c_start:c_end] = outputs[1]
                badpixmetric[:, order, c_start:c_end] = outputs[2]
            else:
                output = pool.apply_async(_extract_flux_chunk, (img_chunk, chunk_locs, chunk_widths, noise_chunk, fit_background))
                pool_jobs.append((output, order, c_start, c_end))
    
    # for multiprocessing, need to retrieve outputs outputs
    if pool is not None:
        for job in pool_jobs:
            job_output, order, c_start, c_end = job
            outputs = job_output.get()
            fluxes[:, order, c_start:c_end] = outputs[0]
            errors[:, order, c_start:c_end] = outputs[1]
            badpixmetric[:, order, c_start:c_end] = outputs[2]
    
    # use the fit metric to mask out bad fluxes
    for fib in range(num_fibers):
        for order in range(num_orders):
            this_metric = badpixmetric[fib, order]
            where_finite_metric = np.where(np.isfinite(this_metric))
            med_val = np.nanmedian(this_metric)
            mad_val = np.nanmedian(np.abs(this_metric - np.nanmedian(this_metric)))
            hist, bin_edges = np.histogram(this_metric[where_finite_metric], bins=np.linspace(-100 * mad_val + med_val, 100 * mad_val + med_val, 200 * 10))
            bin_center = (bin_edges[1::] + bin_edges[0:len(bin_edges) - 1]) / 2.
            # ind = np.argsort(hist)
            # cum_posterior = np.zeros(np.shape(hist))
            cum_posterior = np.cumsum(hist)
            cum_posterior = cum_posterior / np.max(cum_posterior)
            # plt.plot(bin_center,hist/np.max(hist))
            # plt.plot(bin_center,cum_posterior)
            # plt.show()
            rf = interpolate.interp1d(cum_posterior, bin_center, bounds_error=False, fill_value=np.nan)
            upper_bound = rf(1-bad_pixel_fraction)

            where_bad_pixels = np.where((this_metric > upper_bound))

            fluxes[fib, order][where_bad_pixels] = np.nan
            errors[fib, order][where_bad_pixels] = np.nan

    # save the data
    prihdu = fits.PrimaryHDU(data=fluxes, header=img_hdr)
    exthdu = fits.ImageHDU(data=errors)
    hdulist = fits.HDUList([prihdu, exthdu])
    hdulist.writeto(output_filename, overwrite=True)

    return fluxes, errors



def extract_spec(sci_frame, num_sci_frames, noise_frame, trace_centers, trace_sigmas, xs=None, offsets=None):
    """
    Extracts 1-D spectra from the traces

    Args:
        sci_frame: 2-D frame
        num_sci_frame: (int) number of science frames
        noise_frame: 2-D frame of noise per pixel in a frame
        fiber_traces: list of size N_fibers of trace centers arrays with size (N_orders, Nx)
        trace_sigmas: list of size N_fibers of trace sigams arrays with size (Norders, Nx)
    
    Output:
        trace_fluxes: shape of (N_fibers, N_orders, N_x) - 1D extracted fluxes of each fiber
        dark_flxues: shape of (N_offsets, Norders, N_x) - 1D extracted fluxes for each offset
    """

    all_orders = [[] for _ in range(len(trace_centers)+ 1)] # extra one for all the dark traces

    if offsets is None:
        offsets = [23, 15, -15, -23, -31, -39]
    stacked_dark_orders_multi = []

    sci_rotated = np.copy(sci_frame)
    sci_rotated = np.rot90(sci_rotated, -1)

    if xs is None:
        xs = np.arange(10, 1900)

    skynoise = np.rot90(noise_frame)/np.sqrt(num_sci_frames)


    #for trace_params, trace3_params, trace1_params in zip(trace_dat, trace3_dat, trace1_dat):
    for order_index in range(len(trace_centers[0])):
        trace_fits = [fiber_centers[order_index] for fiber_centers in trace_centers]
        sigmas = [fiber_sigmas[order_index] for fiber_sigmas in trace_sigmas]
        
        all_trace_thisorder = [[] for _ in range(len(sigmas) + 1)]
        
        for x in xs:
            #multiple secondary dark
            multi_dark_fluxes = []

            avg_bkgd_flux = []
            for offset in offsets:
                center = trace_fits[0][x] + offset
                center_int = int(np.round(center))
                dat_slice = sci_rotated[center_int-6:center_int+6+1,x]
                ys = np.arange(center_int-6, center_int+6+1)
                sky = skynoise[center_int-6:center_int+6+1,x]

                flux = extract_1d(ys, dat_slice, center, sigmas[0][x], skynoise=sky)
                multi_dark_fluxes.append(flux)
                avg_bkgd_flux.append(dat_slice)
            all_trace_thisorder[-1].append(multi_dark_fluxes)

            bkgd = np.nanmedian(avg_bkgd_flux)

            for this_order, trace_fit, trace_sigma in zip(all_trace_thisorder[:-1], trace_fits, sigmas):
                center = trace_fit[x]
                sigma = trace_sigma[x]
                center_int = int(np.round(center))
                dat_slice = sci_rotated[center_int-6:center_int+6+1,x]
                ys = np.arange(center_int-6, center_int+6+1)
                sky = skynoise[center_int-6:center_int+6+1,x]
                
                flux = extract_1d(ys, dat_slice - bkgd, center, sigma, skynoise=sky)
                #flux = np.nansum(dat_slice)
                this_order.append(flux)
        
         
        for thistrace_thisorder, thistrace_allorders in zip(all_trace_thisorder, all_orders):    
            thistrace_allorders.append(np.array(thistrace_thisorder))
            
    trace_fluxes = np.array(all_orders[:-1])
    dark_fluxes = np.array(all_orders[-1])
    dark_fluxes = np.rollaxis(dark_fluxes, -1)

    return trace_fluxes, dark_fluxes


pix_offsets = np.array([-0.4, -0.2, 0, 0.2, 0.4])
def gauss_cost(params, xs, dat, bkgd):

    flux, sigma, center = params
    #bkgd = 0
    
    fine_xs = np.repeat(xs, 5)
    fine_xs += np.tile(pix_offsets, len(xs))
    
    if sigma < 0:
        return dat
    if bkgd > flux:
        return np.inf * np.ones(dat.shape)
        
    model = flux * np.exp(-(fine_xs - center)**2/(2 * sigma**2)) + bkgd
    
    model = model.reshape(xs.shape[0], 5)
    model = np.mean(model, axis=1)

    return model - dat


def fit_trace(frame, guess_ends, xs=None, plot=False):
    """
    Fit the trace for a fiber using a polynomial for each order

    Args:
        frame: 2-D frame (unrotated))
        guess_ends: a N_orderx2 array, where for each order is a pair of numbers for the endpoints of that order
                    The endpoints are the x coordinate in the unrotated frame, starting with the x coordinate
                    corresponding to the top edge of the trace
        xs: (optiona) the x coordinates in the rotated frame to fit to
        plot: (optiona) plot the extracted flux to sanity check the fit. 
    """
    orders_fluxes = []
    orders_sigmas = []
    orders_sigmas_fit = []
    orders_cuts = []
    orders_centers = []
    orders_center_fit = []

    frame_rot = np.rot90(frame, -1)

    if xs is None:
        xmin = 0
        xmax = frame.shape[0]
        xs = np.arange(xmin, xmax, 1)

    ends = guess_ends

    for end in ends:
        order_fluxes = []
        order_sigmas = []
        order_centers = []
        order_cuts = []
        
        slope = (end[1]-end[0])/frame_rot.shape[1]
        
        y_vals = []
        for x in xs:
            ycen = end[0] + slope*x + 11
            ymin = int(np.round(ycen - 15))
            ymax = int(np.round(ycen + 15))
            
            dat_slice = frame_rot[ymin:ymax+1,x]
            maxind = np.nanargmax(dat_slice) + ymin
            y_vals.append(maxind)
            
        fit = np.polyfit(xs, y_vals, 1)
        fittrace = np.poly1d(fit)
        
        for x in xs:
            maxind = int(np.round(fittrace(x)))
            
            dat_slice = frame_rot[maxind-15:maxind+15+1,x]
            bkgd = np.median([frame_rot[maxind-25:maxind-15+1,x], frame_rot[maxind+15:maxind+25+1,x]])
            
            ys = np.arange(maxind-15, maxind+15+1, dtype=float)

            good = np.where(~np.isnan(dat_slice))
            result = scipy.optimize.leastsq(gauss_cost, (1000, 0.7, maxind), args=(ys[good], dat_slice[good], bkgd))
                                        
            order_fluxes.append(result[0][0] * np.sqrt(2*np.pi) * result[0][1])
            order_sigmas.append(result[0][1])
            order_centers.append(result[0][2])
            order_cuts.append(np.interp(np.arange(-5,6,1) + result[0][2], ys, dat_slice))
            
        order_fluxes = np.array(order_fluxes)
        flux_smooth = ndi.median_filter(order_fluxes, 300)    
        bad = np.where(np.abs(order_fluxes - flux_smooth) > 2 * flux_smooth )
        flux_smooth2 = ndi.median_filter(order_fluxes, 3)    
        order_fluxes[bad] = flux_smooth2[bad]
            
        order_sigma = np.median(order_sigmas)
        sigma_fitargs = np.polyfit(xs, ndi.median_filter(order_sigmas, 15), 1)
        center_fitargs = np.polyfit(xs, ndi.median_filter(order_centers, 15), 3)

        center_fit = np.poly1d(center_fitargs)
        order_centers = center_fit(xs)

        order_sigmas = np.poly1d(sigma_fitargs)(xs)
        
        orders_fluxes.append(order_fluxes)
        orders_sigmas.append(order_sigmas)
        orders_sigmas_fit.append(sigma_fitargs)
        orders_cuts.append(order_cuts)
        orders_centers.append(order_centers)
        orders_center_fit.append(center_fitargs)

    spectral_responses = measure_spectral_response(orders_fluxes)

    if plot:
        import matplotlib.pylab as plt
        fig = plt.figure(figsize=(12,16))
        i = 0
        for order in orders_fluxes:
            ax = fig.add_subplot(9, 1, i+1)
            ax.plot(xs, order,3, 'b-')
            ax.set_ylim([0, np.nanpercentile(order, 99) * 1.2])
            i += 1

        plt.figure(figsize=(16,16))
        plt.imshow(frame_rot, cmap="viridis", interpolation="nearest", vmin=0, vmax=np.nanpercentile(dat_rot, 90))
        plt.gca().invert_yaxis()

        for order_centers in orders_centers:
            plt.plot(xs, order_centers, 'r-' )

        plt.show()

    return orders_centers, orders_sigmas, spectral_responses


def measure_spectral_response(orders_wvs, orders_fluxes, model_wvs, model_fluxes, filter_size=500):
    """
    Measure the spectral response of the instrument/atmosphere (including telluric transmission)

    Args:
        orders_fluxes: Norder x Nchannels array of fluxes per order
        orders_wvs: Norder x Nchannels array of wavelengths per order
        model_wvs: array of wvs of a model spectrum (size N_model)
        model_fluxes: array of fluxes of a model spectrum (size N_model). It's ok if it's not flux normalized. 
        filter_size: (int) smoothing size in pixels
    """

    spectral_responses = []

    for thiswvs, order in zip(orders_wvs, orders_fluxes):
        #continuum = ndi.median_filter(order, filter_size)
        
        model = np.interp(thiswvs, model_wvs, model_fluxes)
        #model_continuum = ndi.median_filter(order, filter_size)

        this_response = order/model
        this_response /= np.nanpercentile(this_response, 99)

        spectral_responses.append(this_response)

    return spectral_responses

def measure_tellurics(orders_fluxes, filter_size=500):
    """
    Measure the high frequency telluric lines emperically from the data. Assume stellar spectrum is featureless

    Args:
        order_fluxes: Norder x Nchannels array of fluxes per order
        filter_size: (int) smoothing size in pixels
    """

    telluric_orders = []

    for order in orders_fluxes:
        star_copy = np.copy(order)
        #star_copy /= np.nanpercentile(star_copy, 99)
        order_response = ndi.median_filter(star_copy, filter_size)
        star_copy /= order_response

        telluric_orders.append(star_copy)

    return telluric_orders



