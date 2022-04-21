import os
import copy as copylib
import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interpolate
import scipy.optimize 
import astropy.io.fits as fits
from astropy.modeling import models, fitting
import astroscrappy
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
import copy
import kpicdrp
import kpicdrp.data as data


gain = kpicdrp.kpic_params.getfloat('NIRSPEC', 'gain')

def process_sci_raw2d(raw_frames, bkgd, badpixmap, detect_cosmics=True, add_baryrv=True, nod_subtraction='none', fiber_goals=None):
    """
    Does simple pre-processing to the raw spectroscopic data: bkgd subtraction, bad pixels, cosmic rays

    Args:
        raw_frames (data.Dataset): dataset of raw detector frames to process
        bkgd (data.Background): 2-D bkgd frame for subtraction
        badpixmap (data.BadPixelMap): 2-D bad pixel map
        detect_cosmics (bool): if True, runs a cosmic ray rejection algorithm on each image
        add_baryrv (bool): If True, add barycentric RV to the header
        nod_subtraction (str): if not 'none', replaces 2-D bkgd calib frame subtraction with nod subtraction with following options:
                                * 'nod': uses adjacent raw_frames on different fibers
                                * 'pair': groups frame sinto pairs and only uses the other frame in a pair
        fiber_goals (list): list of N_frames corresponding to the fiber goal for each frame. Automatically tries to figure out from headers otherwise

    Returns:
        processed_dataset (data.Dataset): a new dataset with basic 2D data processing performed 
    """
    processed_dataset = compute_photon_noise_in_frames(raw_frames)

    processed_dataset = correct_bad_pixels(processed_dataset, badpixmap, detect_cosmics=detect_cosmics, copy=False)

    if add_baryrv:
        processed_dataset = add_baryrv_to_header(processed_dataset, copy=False)

    if nod_subtraction.lower() == 'none':
        processed_dataset = simple_bkgd_subtraction(processed_dataset, bkgd, copy=False)
    else:
        pairsub = nod_subtraction.lower() == 'pair'
        processed_dataset = nod_subtract(processed_dataset, fiber_goals=fiber_goals, pairsub=pairsub, copy=False)

    return processed_dataset


def compute_photon_noise_in_frames(raw_frames, copy=True):
    """
    Computes the noise in each frame assuming all counts are due to photon noise
    Adds them to the noise attirbute

    Args:
        raw_frames (data.Dataset): dataset to modify
        copy (bool): if True, makes copies of input data
    """
    processed_data = []
    for frame in raw_frames:
        if copy:
            new_frame = copylib.deepcopy(frame)
        else:
            new_frame = frame

        var_e = gain * new_frame.data
        noise_e = np.sqrt(var_e)
        noise_adu = noise_e/gain
        new_frame.noise = np.sqrt(new_frame.noise**2 + noise_adu**2)
    
        processed_data.append(new_frame)

    processed_dataset = data.Dataset(processed_data)
    return processed_dataset

def add_baryrv_to_header(frames, copy=True):
    """
    Add barycentric rv to fits headers of extracted frames
    copied from JB's extraction code

    Args:
        frames (data.Dataset): dataset to modify
        copy (bool): if True, makes copies of input data
    """
    processed_data = []
    for frame in frames:
        if copy:
            new_frame = copylib.deepcopy(frame)
            out_header = new_frame.header
        else:
            new_frame = frame
            out_header = frame.header

        keck = EarthLocation.from_geodetic(lat=19.8283 * u.deg, lon=-155.4783 * u.deg, height=4160 * u.m)
        sc = SkyCoord(float(out_header["CRVAL1"]) * u.deg, float(out_header["CRVAL2"]) * u.deg)
        barycorr = sc.radial_velocity_correction(obstime=Time(float(out_header["MJD"]), format="mjd", scale="utc"),
                                                location=keck)
        out_header["BARYRV"] = barycorr.to(u.km / u.s).value

        tnow = Time.now()
        out_header['HISTORY'] = "[{0}] Calculated barycentric RV".format(str(tnow))

        processed_data.append(new_frame)
    
    processed_dataset = data.Dataset(processed_data)

    return processed_dataset


def correct_bad_pixels(raw_frames, badpixmap, detect_cosmics=True, copy=True):
    """
    Performs bad pixel correction based on bad pixel map and looks for cosmic rays

    Args:
        raw_frames (data.Dataset): dataset of raw detector frames to process
        badpixmap (data.BadPixelMap): 2-D bad pixel map
        detect_cosmics (bool): if True, runs a cosmic ray rejection algorithm on each image
        copy (bool): if True, modifies copies of input data
    """
    processed_data = []
    for frame in raw_frames:
        if copy:
            new_hdr = frame.header.copy()
            new_data = np.copy(frame.data)
        else:
            new_hdr = frame.header
            new_data = frame.data

        if detect_cosmics:
            badpixmap4cosmic = np.zeros(badpixmap.data.shape)
            badpixmap4cosmic[np.where(np.isnan(badpixmap.data))] = 1
            dat_crmap, corr_dat = astroscrappy.detect_cosmics(new_data, inmask=badpixmap4cosmic.astype(np.bool))
            new_data[dat_crmap] = np.nan
        new_data[np.where(np.isnan(badpixmap.data))] = np.nan

        tnow = Time.now()
        new_hdr['HISTORY'] = "[{0}] Masked {2} bad pixels using badpixelmap {1}".format(str(tnow), badpixmap.filename, np.size(np.where(np.isnan(badpixmap.data))))

        new_frame = data.DetectorFrame(data=new_data, header=new_hdr, filepath=frame.filepath)

        processed_data.append(new_frame)
    
    processed_dataset = data.Dataset(processed_data)

    return processed_dataset


def simple_bkgd_subtraction(raw_frames, bkgd, scale=False, copy=True):
    """
    Performs simple background subtraciton using a reference background frame. 
        
    Args:
        raw_frames (data.Dataset): dataset of raw detector frames to process
        bkgd (data.Background): 2-D bkgd frame for subtraction
        pairsub (bool): if True, groups frames into pairs and does subtraction, otherwise looks at both adjacent pairs
        copy (bool): if True, modifies copies of input data

    """
    processed_data = []
    for frame in raw_frames:
        if copy:
            new_hdr = frame.header.copy()
        else:
            new_hdr = frame.header

        if scale:
            scale_factor = (np.nanmedian(frame.data)/np.nanmedian(bkgd.data))
        else:
            scale_factor = 1

        new_data = frame.data - (bkgd.data * scale_factor)
        new_noise = np.sqrt(frame.noise**2 + bkgd.noise**2)

        tnow = Time.now()
        new_hdr['HISTORY'] = "[{0}] Subtracted thermal background frame {1}".format(str(tnow), bkgd.filename)

        new_frame = data.DetectorFrame(data=new_data, header=new_hdr, filepath=frame.filepath, noise=new_noise)
        new_frame.filename = new_frame.filename[:-5] + "_bkgdsub.fits"


        processed_data.append(new_frame)
    
    processed_dataset = data.Dataset(processed_data)

    return processed_dataset

def nod_subtract(raw_frames, fiber_goals=None, pairsub=False, copy=True):
    """
    Performs nod subtraction by using adjacent frames 

    Args:
        raw_frames (data.Dataset): dataset of raw detector frames to process
        fiber_goals (list): list of N_frames corresponding to the fiber goal for each frame
        pairsub (bool): if True, groups frames into pairs and does subtraction, otherwise looks at both adjacent pairs
        copy (bool): if True, modifies copies of input data
    """ 
    if fiber_goals is None:
        fiber_goals = raw_frames.get_header_values("FIUGNM")
    
    # group frames by fiber goal
    group_index = 0
    fiber_groups = np.zeros(len(fiber_goals))
    curr_group = fiber_goals[0]
    for i in range(len(raw_frames)):
        if fiber_goals[i] != curr_group:
            group_index += 1
            curr_group = fiber_goals[i]
        fiber_groups[i] = group_index
    
    processed_data = []
    for i in range(len(raw_frames)):
        # figure out which frames to use
        if not pairsub:
            good_frames = np.where(np.abs(fiber_groups - fiber_groups[i]) == 1)
        else:
            if fiber_groups[i] % 2 == 0:
                pair_index = fiber_groups[i] + 1
                # if somehow there is an odd number of groups, make the last pair as an exception
                if pair_index > fiber_groups.max():
                    pair_index -= 2
            else:
                pair_index = fiber_groups[i] - 1
            good_frames = np.where(fiber_groups == pair_index)
        
        bkgd = np.nanmean(raw_frames[good_frames].get_dataset_attributes('data'), axis=0)
        bkgd_noise = np.nanmean(raw_frames[good_frames].get_dataset_attributes('noise'), axis=0)/np.sqrt(np.size(good_frames))
        files_used = raw_frames[good_frames].get_dataset_attributes('filename')

        new_frame = raw_frames[i].data - bkgd
        new_noise = np.sqrt(raw_frames[i].noise**2 + bkgd_noise**2)

        if copy:
            output_hdr = raw_frames[i].header.copy()
        else:
            output_hdr = raw_frames[i].header

        tnow = Time.now()
        new_frame = data.DetectorFrame(data=new_frame, header=output_hdr, filepath=raw_frames[i].filepath, noise=new_noise)
        new_frame.filename = new_frame.filename[:-5] + "_nodsub.fits"
        new_frame.header['HISTORY'] = "[{0}] Performed nod background subtraction with the following frames: {1}".format(str(tnow), " ".join(files_used))
        
        processed_data.append(new_frame)

    processed_dataset = data.Dataset(processed_data)

    return processed_dataset


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
    g = models.Gaussian1D(amplitude=1./np.sqrt(2*np.pi*sigma**2), mean=center, stddev=sigma)
    good = np.where(~np.isnan(dat_slice))
    if np.size(good) < 3:
        return np.nan, np.nan, np.nan, np.nan
    good_slice = dat_slice[good]
    good_coords = dat_coords[good]
    noise = noise[good]
    initial_flux = np.sum(good_slice * g(good_coords))/np.sum(g(good_coords)*g(good_coords))
    
    good_var = noise**2
    bkgd_var = noise**2 # TODO: not currently implemented. 
    flux = np.sum(good_slice * g(good_coords) / good_var)/np.sum(g(good_coords)*g(good_coords)/good_var)
    flux_err = np.sqrt(1./np.sum(g(good_coords)*g(good_coords)/good_var))
    flux_err_bkgd_only = np.sqrt(1./np.sum(g(good_coords)*g(good_coords)/bkgd_var))
    residuals = flux * g(good_coords) - good_slice
    max_res = np.nanmax(np.abs(residuals))
    res_mad = np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))
    badpixmetric = np.abs(max_res)
    # badpixmetric = np.abs(max_res/res_mad)

    # if flux > 200:
    #     import matplotlib.pylab as plt
    #     plt.plot(good_coords, good_slice, 'o')
    #     plt.show()
    #     import pdb; pdb.set_trace()
    #if np.isnan(flux):
    #    import pdb; pdb.set_trace()
    
    return flux, flux_err, flux_err_bkgd_only, max_res

def extract_1d_box(dat_coords, dat_slice, center, sigma, noise):
    """
    Box extraction of a single 1-D slice

    Args:
        dat_coords: y-coordinates of dat_slice (Ny)
        dat_slice: data along a slice (Ny)
        center: center of Gaussian in dat_coords coordinates (float)
        sigma: standard deviation of Gaussian (float)
        noise: noise in each pixel of data (Ny)

    Returns:
        flux: total integrated flux in the box
        badpixmetric metric for how good of a flux estimate this is (float)
    """
    g = models.Gaussian1D(amplitude=1./np.sqrt(2*np.pi*sigma**2), mean=center, stddev=sigma)
    good = np.where(~np.isnan(dat_slice))
    if np.size(good) < 3:
        return np.nan, np.nan, np.nan, np.nan
    good_slice = dat_slice[good]
    good_coords = dat_coords[good]
    noise = noise[good]
    flux = np.sum(good_slice)
    flux_err = np.sqrt(np.sum(noise**2+good_slice/gain))
    flux_err_bkgd_only = np.sqrt(np.sum(noise**2))
    max_res = np.nan ### This isn't meaningful for a box extraction

    return flux, flux_err, flux_err_bkgd_only, max_res


def _extract_flux_chunk(image, order_locs, order_widths, img_noise, fit_background, sci_fibers, box=False):
    """
    Extracts the flux from a chunk of size Nx of an order for some given fibers

    Args:
        image: 2-D frame (Ny, Nx)
        order_locs: y-location of fiber traces in an order (N_fibers, Nx)
        order_widths: standard deviation of fiber traces in an order (N_fibers, Nx)
        img_noise: 2-D frame that describes the thermal/background noise in each pixel (Ny, Nx). if None, will try to compute this emperically
            The added Poisson noise from the signal is added to this noise map within the function so it should not be included in this input map.
        fit_background: if True, fit the background
        sci_fibers: index of fibers (up to N_fibers) that are science fibers 

    Returns:
        fluxes: (N_fibers, Nx) extracted 1-D fluxes
        fluxerrs_extraction: (N_fibers, Nx) errors for 1-D extracted fluxes due to extraction
        fluxerrs_bkgd_only: (N_fibers, Nx) background only component of the flux errors
        fluxerrs_emperical: (N_fibers, Nx) emperical attempt to estimate flux errors
        badpixmetric: (N_fibers, Nx) metric for how bad a pixel is 
    """
    fluxes = np.zeros(order_locs.shape)+np.nan
    fluxerrs_extraction = np.zeros(order_locs.shape)+np.nan
    fluxerrs_bkgd_only = np.zeros(order_locs.shape)+np.nan
    fluxerrs_emperical = np.zeros(order_locs.shape)+np.nan
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
        # for each science fiber, mask 11 pixels around it
        for center in centers[sci_fibers]:
            center_int = int(np.round(center))
            bkgd_column[center_int-5:center_int+5+1] = np.nan 
        ymin_long = int(np.round(np.min(order_locs[sci_fibers, x]))) - 6
        ymax_long = int(np.round(np.max(order_locs[sci_fibers, x]))) + 6 + 1
        bkgd_slice = bkgd_column[ymin_long:ymax_long]
        bkgd_noise = np.nanstd(bkgd_slice) # roughly the noise
        num_bkgd_points = np.size(bkgd_slice[np.where(~np.isnan(bkgd_slice))])

        if fit_background:
            # compute background as median of remaining pixels
            bkgd_level = np.nanmedian(bkgd_slice)
            err_bkgd = bkgd_noise/np.sqrt(num_bkgd_points) * (np.pi/2) # pi/2 factor is due to using median instead of mean. 
        else:
            bkgd_level = 0
            err_bkgd = 0

        column_maxres = []
        for fiber in range(num_fibers):
            center = order_locs[fiber, x]
            sigma = order_widths[fiber, x]
            # if either is nan, we can't do anything about the extraction. 
            if (np.isnan(center)) or (np.isnan(sigma)):
                fluxes[fiber, x] = np.nan
                fluxerrs_extraction[fiber, x] = np.nan
                fluxerrs_bkgd_only[fiber, x] = np.nan
                fluxerrs_emperical[fiber, x] = np.nan
                continue          

            center_int = int(np.round(center))

            # JX: when would this be nan? on bad pixels?
            if center < 6 or center >= np.size(img_column)-6:
                flux, flux_err_extraction, flux_err_bkgd_only, maxres = np.nan, np.nan, np.nan, np.nan
            elif np.any(np.isnan(img_column[center_int-1:center_int+2])):
                flux, flux_err_extraction, flux_err_bkgd_only, maxres = np.nan, np.nan, np.nan, np.nan
            else:
                # take a slice of data and subtract the background
                # JX: Why 6?
                dat_slice = img_column[center_int-6:center_int+6+1] - bkgd_level
                ys = np.arange(center_int-6, center_int+6+1)
                noise = np.sqrt(img_noise[center_int-6:center_int+6+1, x]**2 + err_bkgd**2)
                noise[np.where(np.isnan(noise))] = np.sqrt(bkgd_noise**2 + err_bkgd**2)

                #flux, badpixmetric = extract_1d(ys, dat_slice, center, sigma, noise)
                if box: 
                    flux, flux_err_extraction, flux_err_bkgd_only, maxres = extract_1d_box(ys, dat_slice, center, sigma, noise)
                else:
                    flux, flux_err_extraction, flux_err_bkgd_only, maxres = extract_1d(ys, dat_slice, center, sigma, noise)

            column_maxres.append(maxres)
            fluxes[fiber, x] = flux
            # account for the fact we are doing total integrated flux of Gaussian when computing the noise
            # JX: sometimes flux is too negative and makes arg of sqrt negative, throwing Runtime warning
            fluxerr_emperical = np.sqrt(gain*flux + (bkgd_noise * np.sqrt(2*np.pi) * sigma)**2 )
            fluxerrs_extraction[fiber, x] = np.sqrt(flux_err_extraction**2 + err_bkgd**2) 
            fluxerrs_bkgd_only[fiber, x] = flux_err_bkgd_only
            fluxerrs_emperical[fiber, x] = fluxerr_emperical

        column_maxres = np.array(column_maxres)    
        
        # ymin_long = int(np.round(np.min(order_locs[:, x]))) - 6
        # ymax_long = int(np.round(np.max(order_locs[:, x]))) + 6 + 1
        # ys_long = np.arange(ymin_long, ymax_long)
        # long_slice = img_column[ymin_long:ymax_long] - bkgd_level
        # for fiber in range(num_fibers):
        #     sigma = order_widths[fiber, x]
        #     peakflux = fluxes[fiber, x]/(np.sqrt(2*np.pi) * sigma)
        #     g = models.Gaussian1D(amplitude=peakflux, mean=order_locs[fiber, x], stddev=sigma)
        #     long_slice -= g(ys_long)

        # these_badmetrics = column_maxres/np.nanstd(long_slice)#/np.nanmedian(np.abs(long_slice - np.nanmedian(long_slice)))
        these_badmetrics = column_maxres
        badpixmetrics[:, x] = these_badmetrics

    return fluxes, fluxerrs_extraction, fluxerrs_bkgd_only, fluxerrs_emperical, badpixmetrics
        


def extract_flux(dataset, trace_params, fit_background=False, bad_pixel_fraction=0.0, pool=None, box=False):
    """
    Extracts the flux from the traces to make 1-D spectra. 

    Args:
        dataset (data.Dataset): dataset of DetectorFrame images that we want to extract fluxes from
        trace_params (data.TraceParams): trace parameters to use
        fit_background: if True, fits for a constant background during trace extraction (default: False). 
        trace_flags: numberical flags array of length N_fibers.
            0 indicates regular fiber.
            1 indicates background fictitious fiber on the slit trace.
            2 indicates background fictitious fiber outside the slit trace.
        bad_pixel_fraction: assume this fraction of all 1-D fluxes are bad and will be masked as nans (default 0.01)
        pool: optional multiprocessing.Pool object to pass in to parallelize the spectral extraction (default: None)

    Return:
        Saved also in output_filename as a fits file:
        fluxes: (N_fibers, N_orders, Nx) extracted 1-D fluxes. 
        errors: (N_fibers, N_orders, Nx) errors on extracted 1-D fluxes.
    """

    trace_flags = trace_params.labels
    sci_fibers = trace_params.get_sci_indices()
    trace_widths = trace_params.widths
    trace_locs = trace_params.locs
    num_fibers = trace_locs.shape[0]
    num_orders = trace_locs.shape[1]
    nx = trace_locs.shape[2]
    # break up image into chunks
    chunk_size = 8
    num_chunks = nx // 8


    spectral_data = []

    for frame in dataset:
        image = frame.data
        img_noise = frame.noise
        # if the frame as no noise, mark is as nan
        if np.all(img_noise == 0):
            img_noise = np.zeros(image.shape) * np.nan

        fluxes = np.zeros(trace_locs.shape)+np.nan
        fluxerrs_extraction = np.zeros(trace_locs.shape)+np.nan
        fluxerrs_bkgd_only = np.zeros(trace_locs.shape)+np.nan
        fluxerrs_emperical = np.zeros(trace_locs.shape)+np.nan
        badpixmetrics = np.zeros(trace_locs.shape)+np.nan

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
                    outputs = _extract_flux_chunk(img_chunk, chunk_locs, chunk_widths, noise_chunk, fit_background, sci_fibers, box=box)
                    fluxes[:, order, c_start:c_end] = outputs[0]
                    fluxerrs_extraction[:, order, c_start:c_end] = outputs[1]
                    fluxerrs_bkgd_only[:, order, c_start:c_end] = outputs[2]
                    fluxerrs_emperical[:, order, c_start:c_end] = outputs[3]
                    badpixmetrics[:, order, c_start:c_end] = outputs[-1]
                else:
                    output = pool.apply_async(_extract_flux_chunk, (img_chunk, chunk_locs, chunk_widths, noise_chunk, fit_background, sci_fibers), dict(box=box))
                    pool_jobs.append((output, order, c_start, c_end))

        # for multiprocessing, need to retrieve outputs outputs
        if pool is not None:
            for job in pool_jobs:
                job_output, order, c_start, c_end = job
                outputs = job_output.get()
                fluxes[:, order, c_start:c_end] = outputs[0]
                fluxerrs_extraction[:, order, c_start:c_end] = outputs[1]
                fluxerrs_bkgd_only[:, order, c_start:c_end] = outputs[2]
                fluxerrs_emperical[:, order, c_start:c_end] = outputs[3]
                badpixmetrics[:, order, c_start:c_end] = outputs[-1]
        
        # use the fit metric to mask out bad fluxes
        for fib in range(num_fibers):
            for order in range(num_orders):
                this_metric = badpixmetrics[fib, order]
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
                fluxerrs_extraction[fib, order][where_bad_pixels] = np.nan
                fluxerrs_bkgd_only[fib, order][where_bad_pixels] = np.nan
                fluxerrs_emperical[fib, order][where_bad_pixels] = np.nan


        fileprefix = frame.filename[:-5] + "_spectra.fits"
        filepath = os.path.join(frame.filedir, fileprefix)
        this_spec = data.Spectrum(fluxes=fluxes, errs=fluxerrs_extraction, header=frame.header, filepath=filepath, labels=trace_params.labels)
        spectral_data.append(this_spec)

    spectral_data = np.array(spectral_data)
    spectral_dataset = data.Dataset(frames=spectral_data)

    # # save the data
    # if output_filename is not None:
    #     out_hdr = add_baryrv(img_hdr)
    #     prihdu = fits.PrimaryHDU(data=fluxes[np.where(trace_flags==0)], header=out_hdr)
    #     exthdu1 = fits.ImageHDU(data=fluxerrs_extraction[np.where(trace_flags==0)],)
    #     exthdu2 = fits.ImageHDU(data=fluxes[np.where(trace_flags==1)],)
    #     exthdu3 = fits.ImageHDU(data=fluxes[np.where(trace_flags==2)],)
    #     exthdu4 = fits.ImageHDU(data=fluxerrs_bkgd_only[np.where(trace_flags==0)],)
    #     hdulist = fits.HDUList([prihdu, exthdu1, exthdu2, exthdu3,exthdu4])
    #     hdulist.writeto(output_filename, overwrite=True)

    return spectral_dataset




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

    model_spectra = np.array([np.interp(thiswvs, model_wvs, model_fluxes) for thiswvs in orders_wvs])

    spectral_responses = orders_fluxes / model_spectra
    spectral_responses /= np.nanpercentile(spectral_responses, 99)

    # for thiswvs, order in zip(orders_wvs, orders_fluxes):
    #     #continuum = ndi.median_filter(order, filter_size)
        
    #     model = np.interp(thiswvs, model_wvs, model_fluxes)
    #     #model_continuum = ndi.median_filter(order, filter_size)

    #     this_response = order/model
    #     order_norm = np.nanpercentile(this_response, 99)
    #     this_response /= np.nanpercentile(this_response, 99)

    #     spectral_responses.append(this_response)

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



