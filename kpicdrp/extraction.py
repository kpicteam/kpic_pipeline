import numpy as np
import scipy.ndimage as ndi
import scipy.optimize
import astropy.io.fits as fits
from astropy.modeling import models, fitting
import astroscrappy
import  astropy.time as time
import astropy.units as u
from astropy.coordinates import SkyCoord

gain = 2.85 # e-/ADU

def process_bkgds(filelist, detect_cosmics=True):
    """
    Makes a master bkgd frame from a series of background frames

    Args:
        filelist: list of filenames to read in
    Returns:
        master_bkgd: the combined background frame
        bkgd_noise: the noise associated with each pixel 
        bkgd_frames: the indivudal frames after badpix and cosmic ray rejection
    """
    master_bkgd = []

    for filename in filelist:
        with fits.open(filename) as hdulist:
            bkgd = np.copy(hdulist[0].data)
            
            if detect_cosmics:
                crmap, _ = astroscrappy.detect_cosmics(bkgd)
                bkgd[crmap] = np.nan
            
            master_bkgd.append(bkgd)
        hdulist.close()

    bkgd_frames = master_bkgd
    bkgd_noise = np.nanstd(master_bkgd, axis=0)
    master_bkgd = np.nanmean(master_bkgd, axis=0)

    badpixmap = np.zeros(master_bkgd.shape)
    badpixmap[np.where(np.isnan(master_bkgd))] = 1
    
    for i in range(2):
        smoothed_thermal_noise = ndi.median_filter(bkgd_noise, 3)
        smoothed_thermal_noise_percolumn = np.nanmedian(smoothed_thermal_noise, axis=0)
        smoothed_thermal_noise_percolumn_2d = np.ones(bkgd_noise.shape) * smoothed_thermal_noise_percolumn[None, :]

        smoothed_thermal_noise[np.where(badpixmap == 1)] = smoothed_thermal_noise_percolumn_2d[np.where(badpixmap == 1)]

        master_bkgd_smooth = ndi.median_filter(master_bkgd, 7)

        # radius = 3
        # mean = ndi.uniform_filter(master_dark, radius*2, mode='constant', origin=-radius)
        # c2 = ndi.uniform_filter(master_dark*master_dark, radius*2, mode='constant', origin=-radius)
        # master_dark_std = ((c2 - mean*mean)**.5)
        badpixmap[np.where( (np.abs(master_bkgd - master_bkgd_smooth) > 6 * smoothed_thermal_noise/np.sqrt(12)) | (master_bkgd > 1000)) ] = 1

        badpix = np.where(badpixmap == 1)

        master_bkgd[badpix] = np.nan
        bkgd_noise[badpix] = np.nan

    smoothed_thermal_noise_percolumn = np.nanmedian(smoothed_thermal_noise, axis=0)
    smoothed_thermal_noise_percolumn_2d = np.ones(bkgd_noise.shape) * smoothed_thermal_noise_percolumn[None, :]
    bad_thermal = np.where(np.isnan(smoothed_thermal_noise))
    smoothed_thermal_noise[bad_thermal] = smoothed_thermal_noise_percolumn_2d[bad_thermal]

    return master_bkgd, smoothed_thermal_noise, badpixmap

def process_sci_raw2d(filelist, bkgd, badpixmap, detect_cosmics=True):
    """
    Does simple processing to the raw spectroscopic data: bkgd subtraction, bad pixels, cosmic rays

    Args:
        filelist: list of science files to process
        bkgd: 2-D bkgd frame for subtraction
        badpixmap: 2-D bad pixel map
    """

    sci_data = []

    for filename in filelist:
        with fits.open(filename) as hdulist:
            dat = np.copy(hdulist[0].data)
            dat -= bkgd
            
            if detect_cosmics:
                dat_crmap, corr_dat = astroscrappy.detect_cosmics(dat, inmask=badpixmap)
                dat[dat_crmap] = np.nan
            dat[np.where(badpixmap == 1)] = np.nan
            
            sci_data.append(dat)
        hdulist.close()

    sci_noise = np.nanstd(sci_data, axis=0)
    sci_frames = sci_data
    sci_data = np.nanmean(sci_data, axis=0)

    return sci_data, sci_noise, sci_frames


def extract_1d(dat_coords, dat_slice, center, sigma, skynoise):
    """
    Optimal (I think) extraction of a single 1-D slice 
    """
    g = models.Gaussian1D(amplitude=1, mean=center, stddev=sigma)
    good = np.where(~np.isnan(dat_slice))
    good_slice = dat_slice[good]
    good_coords = dat_coords[good]
    skynoise = skynoise[good]
    initial_flux = np.sum(good_slice * g(good_coords))/np.sum(g(good_coords)*g(good_coords))
    
    good_var = (initial_flux * g(good_coords) + skynoise**2)/gain
    flux = np.sum(good_slice * g(good_coords) / good_var)/np.sum(g(good_coords)*g(good_coords)/good_var)
    
    flux *= np.sqrt(2*np.pi) * sigma
    
    #if np.isnan(flux):
    #    import pdb; pdb.set_trace()
    
    return flux



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

