import numpy as np
import scipy.ndimage as ndi
import astropy.units as u
import astropy.constants as consts



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
