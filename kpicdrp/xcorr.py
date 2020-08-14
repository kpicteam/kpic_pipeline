import numpy as np
import scipy.ndimage as ndi
import astropy.units as u
import astropy.constants as consts
import scipy.optimize as optimize
import scipy.interpolate as interp
from PyAstronomy import pyasl



def simple_xcorr(shifts, orders_wvs, orders_fluxes, template_wvs, template_fluxes, telluric_wvs=None, telluric_fluxes=None, orders_responses=None):
    """
    Do a simple CCF 

    Args:
        shifts: np.array of shifts in km/s
        orders_wvs: (Norders, Nchannels) array of wavelengths
        orders_fluxes: (Norders, Nchannels) array of fluxes
        template_wvs: np.array of wvs for the template
        template_fluxes: np.array of fluxes for teh template
        telluric_wvs: np.array of wvs for the telluric model
        telluric_fluxes: npm.array of fluxes for the telluric model
        orders_responses (Norders, Nchannels) array of spectral responses
    """

    total_xcorrs = []
    total_model_xcorrs = []
    model_var = []
    data_var = []
    model_noshift_var = []


    for i in range(orders_wvs.shape[0]):    
        thiswvs = orders_wvs[i]
        order = orders_fluxes[i]
        mad = np.ones(order.shape)
        #star_order = cleaned_star_orders[i]
        
        if telluric_wvs is not None:
            tell_template = np.interp(thiswvs, telluric_wvs, telluric_fluxes)
        else:
            tell_template = 1
        
        if orders_responses is not None:
            resp_template = orders_responses[i]
        else:
            resp_template = 1

        bad = np.where(np.isnan(order) | (tell_template < 0.50) )
        
        order_copy = np.copy(order)
        order_mask = np.where(np.isnan(order))
        order_copy[order_mask] = np.nanmedian(order)
        order_copy -= ndi.median_filter(order_copy, 200)
        order_copy[bad] = np.nan

        xcorrs = []
        model_xcorrs = []
        order_model_var = []
        order_data_var = []
        order_model_noshift_var = []
        for shift in shifts:
            new_beta = shift/consts.c.to(u.km/u.s).value 
            new_redshift = np.sqrt((1 + new_beta)/(1 - new_beta)) - 1
            wvs_starframe = thiswvs/(1+new_redshift)

            template = np.interp(wvs_starframe, template_wvs, template_fluxes) 
            template_noshift = np.interp(thiswvs, template_wvs, template_fluxes) 

            template *= resp_template * tell_template 
            template_noshift *= resp_template * tell_template 
            template[order_mask] = np.nanmedian(template)
            template_noshift[order_mask] = np.nanmedian(template_noshift)
            template /= ndi.median_filter(template, 200)
            template_noshift /= ndi.median_filter(template_noshift, 200)
            
            xcorr = (template * order_copy)
            model_xcorr = (template * template_noshift)
            
            xcorrs.append(xcorr)
            model_xcorrs.append(model_xcorr)
            order_model_var.append((template * template))
            order_data_var.append((order_copy * order_copy))
            order_model_noshift_var.append((template_noshift * template_noshift))
            
        xcorrs = np.array(xcorrs)
        model_xcorrs = np.array(model_xcorrs)
        
        total_xcorrs.append(xcorrs)
        total_model_xcorrs.append(model_xcorrs)
        model_var.append(order_model_var)
        data_var.append(order_data_var)
        model_noshift_var.append(order_model_noshift_var)
        


    model_var = np.array(model_var)
    data_var = np.array(data_var)
    model_noshift_var = np.array(model_noshift_var)

    ccf = np.nanmean(total_xcorrs, axis=(0,2))/np.sqrt(np.nanmean(model_var, axis=(0,2)) * np.nanmean(data_var, axis=(0,2)))
    acf = np.nanmean(total_model_xcorrs, axis=(0,2))/np.sqrt(np.nanmean(model_var, axis=(0,2)) * np.nanmean(model_noshift_var, axis=(0,2)))

    return ccf, acf


def generate_forward_model_singleorder(fitparams, orders_wvs, order_sigmas, star_wvs, star_template_fluxes, template_wvs, template_fluxes, orders_responses, broadened=False):
    rvshift, vsini, pl_flux, star_flux = fitparams


    if not broadened:
        if vsini < 0:
            # bad!
            broad_model = np.ones(template_fluxes.shape)
        else:  
            broad_model = pyasl.rotBroad(template_wvs, template_fluxes, 0.1, vsini)
    else:
        broad_model = template_fluxes



    thiswvs = orders_wvs
    star_template = np.interp(thiswvs, star_wvs, star_template_fluxes)
    star_template /= np.percentile(star_template, 90)


    new_beta =(rvshift)/consts.c.to(u.km/u.s).value #+  (rel_v)/consts.c.to(u.km/u.s).value
    new_redshift = np.sqrt((1 + new_beta)/(1 - new_beta)) - 1

    template_wvs_starframe = template_wvs/(1+new_redshift) 

    template = np.interp(template_wvs, template_wvs_starframe, broad_model)
    template /= np.percentile(template, 90)

    # broaden to instrumental resolution 
    template = convolve_and_interp(thiswvs, order_sigmas, template_wvs, broad_model)
    # model_r = np.median(template_wvs/np.median(template_wvs - np.roll(template_wvs, 1)))
    # data_r = 35000
    # downsample = model_r/data_r/(2*np.sqrt(2*np.log(2)))
    # broad_model = ndi.gaussian_filter(broad_model, downsample)

    resp_template = orders_responses


    template *= resp_template 

    template *= pl_flux
    star_template *= star_flux

    template += star_template


    return template



def grid_search(orders_wvs, orders_fluxes, orders_fluxerrs, star_wvs, star_template_fluxes, template_wvs, template_fluxes, orders_responses):
        
    loglikes = []

    shifts = np.linspace(-80, 20, 20)
    broadening = np.linspace(1, 50, 15)
    contrasts = np.linspace(1, 40, 20)
    star_fluxes = np.linspace(450, 320, 20)

    for vsini in broadening:
        print("vsini", vsini)
    #         model_in_band = np.where((L1_dat['wvs'] >= np.min(thiswvs)) & (L1_dat['wvs'] <= np.max(thiswvs)))
    #         model_dwv = np.abs(np.median(np.roll(L1_dat['wvs'][model_in_band], 1) - L1_dat['wvs'][model_in_band]))

        broad_model = pyasl.rotBroad(template_wvs, template_fluxes, 0.1, vsini)
        #broad_model = resampled_model_flux
        
        shift_xcorrs = []
        for shift in shifts:
            print("RV shift", shift)

            contrast_xcorrs = []
            
            for contrast in contrasts:
                sflux_xcorrs = []
                
                for star_flux in star_fluxes:
                    order_xcorrs = []

                    orders = [2]
                    fitparams = (shift, vsini, contrast, star_flux)
                    model_orders = generate_forward_model_singleorder(fitparams, orders_wvs[orders], star_wvs, star_template_fluxes, template_wvs, broad_model, orders_responses[orders], broadened=True)

                    model_continuun = ndi.median_filter(model_orders, 100)
                    data_continuum = ndi.median_filter(orders_fluxes[orders], 100)

                    norm_model = model_orders - model_continuun + np.median(model_continuun)
                    norm_data = orders_fluxes[orders] - data_continuum + np.median(data_continuum)
                    norm_errs = orders_fluxerrs[orders] / data_continuum

                    this_loglike = -0.5 * (norm_model - norm_data)**2/norm_errs**2

                    sflux_xcorrs.append(np.nansum(this_loglike))
                    
                contrast_xcorrs.append(sflux_xcorrs)
                
            shift_xcorrs.append(contrast_xcorrs)
            
        loglikes.append(shift_xcorrs) 

    loglikes = np.array(loglikes)
    return loglikes


def lsqr_fit(guess, orders, orders_wvs, orders_sigmas, orders_fluxes, orders_fluxerrs, star_wvs, star_template_fluxes, template_wvs, template_fluxes, orders_responses):
                   
    all_norm_data = []
    all_norm_errs = []

    # normalize the data
    for data, errs in zip(orders_fluxes[orders], orders_fluxerrs[orders]):
        bad = np.where(np.isnan(data))
        data_copy = np.copy(data)
        data_copy[bad] = np.nanmedian(data)
        data_continuum = ndi.median_filter(data_copy, 100)
        norm_data = data - data_continuum + np.nanmedian(data_continuum)
        norm_errs = errs #/ data_continuum

        all_norm_data.append(norm_data)
        all_norm_errs.append(norm_errs)

    def cost_function(fitparams):
        shift, vsini, = fitparams[0:2]

        all_diffs = []
        broad_model = pyasl.rotBroad(template_wvs, template_fluxes, 0.1, vsini)



        for i, order in enumerate(orders):
            contrast, star_flux = fitparams[2*i+2:2*i+4]

            this_fitparams = [shift, vsini, contrast, star_flux]

            model_orders = generate_forward_model_singleorder(this_fitparams, orders_wvs[order], orders_sigmas[order], star_wvs, star_template_fluxes, template_wvs, broad_model, orders_responses[order], broadened=True)
            
            good = np.where(~np.isnan(model_orders))
            model_orders = model_orders[good]
            
            model_continuum = ndi.median_filter(model_orders, 100)
            norm_model = model_orders - model_continuum + np.nanmedian(model_continuum)

            norm_data = all_norm_data[i][good]
            norm_errs = all_norm_errs[i][good]

            diff = (norm_model - norm_data)/norm_errs
            all_diffs = np.append(all_diffs, diff[np.where(~np.isnan(diff))])
        
        return all_diffs

    bounds_lower = [-150 , 0]
    bounds_upper = [150, 100]
    for i in range(len(orders)):
        bounds_lower += [0, 0]
        bounds_upper += [np.inf, np.inf]
        guess += [10, 100]

    result = optimize.least_squares(cost_function, guess, bounds=(bounds_lower, bounds_upper))

    return result


def convolve_and_interp(wv_channels, sigmas, model_wvs, model_fluxes, channel_width=None, num_sigma=3):
    """
    Simulate the observations of a model. Convolves the model with a variable Gaussian LSF in each spectral channel.

    Args:
        wv_channels: the wavelengths desired (length of N_output)
        sigmas: the LSF gaussian stddev of each wv_channels in units of channels (length of N_output)
        model_wvs: the wavelengths of the model (length of N_model)
        model_fluxes: the fluxes of the model (length of N_model)
        channel_width: (optional) the full width of each wavelength channel in units of wavelengths (length of N_output)
        num_sigma (float): number of +/- sigmas to evaluate the LSF to. 

    Returns:
        output_model: the fluxes in each of the wavelength channels (length of N_output)
    """
    # create the wavelength grid for variable LSF convolution
    dwv = np.abs(wv_channels - np.roll(wv_channels, 1))
    dwv[0] = dwv[1] # edge case
    sigmas_wvs = sigmas * dwv
    #filter_size_wv = int(np.ceil(np.max(sigmas_wvs))) * 6 # wavelength range to filter

    model_in_range = np.where((model_wvs >= np.min(wv_channels)) & (model_wvs < np.max(wv_channels)))
    dwv_model = np.abs(model_wvs[model_in_range] - np.roll(model_wvs[model_in_range], 1))
    dwv_model[0] = dwv_model[1]

    filter_size = int(np.ceil(np.max((2 * num_sigma * sigmas_wvs)/np.min(dwv_model)) ))
    filter_coords = np.linspace(-num_sigma, num_sigma, filter_size)
    filter_coords = np.tile(filter_coords, [wv_channels.shape[0], 1]) #  shape of (N_output, filter_size)
    filter_wv_coords = filter_coords * sigmas_wvs[:,None] + wv_channels[:,None] # model wavelengths we want
    lsf = np.exp(-filter_coords**2/2)

    model_interp = interp.interp1d(model_wvs, model_fluxes, kind='cubic', bounds_error=False)
    filter_model = model_interp(filter_wv_coords)

    output_model = np.nansum(filter_model * lsf, axis=1)/np.sum(lsf * lsf, axis=1)
    
    return output_model
