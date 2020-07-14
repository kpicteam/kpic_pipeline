import numpy as np
import scipy.ndimage as ndi
import scipy.optimize
import astropy.io.fits as fits
import astropy.io.ascii as ascii
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import astropy.constants as consts
import astropy.table as table
import matplotlib.pylab as plt
from astropy.modeling import models, fitting

from astropy.utils import iers
from astropy.utils.iers import conf as iers_conf
print(iers_conf.iers_auto_url)
#default_iers = iers_conf.iers_auto_url
#print(default_iers)
iers_conf.iers_auto_url = 'https://datacenter.iers.org/data/9/finals2000A.all'
iers_conf.iers_auto_url_mirror = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'
iers.IERS_Auto.open()  # Note the URL


def fit_to_model(fitparams, xs, data, redshift, tell_wv, tell_trans, template_wvs, template_flux, response_xs, response_flux):
    
    #a2 = fitparams[0]
    #a1 = fitparams[1]
    #a0 = fitparams[2]
    polyparams = fitparams
    params = [polyparams[0]/1e25, polyparams[1]/1e20, polyparams[2]/1e15, polyparams[3]/1e10, polyparams[4]/1e5, polyparams[5]]

    xcoords = xs
    #wvs_model = a2 * xcoords**2 + a1 * xcoords + a0
    wvs_model = np.poly1d(params)(xcoords)
    
    #tell_trans = ndi.gaussian_filter(tell_trans, 3)

    template_interp = np.interp(wvs_model, template_wvs*(1+redshift), template_flux)
    tell_interp = np.interp(wvs_model, tell_wv, tell_trans)
    response_interp = np.interp(xcoords, response_xs, response_flux)

    model = template_interp *  response_interp  * tell_interp
    #model *= np.percentile(data, 95)/np.percentile(model, 95)
    model /= ndi.median_filter(model, 300)
    data_filt = data / ndi.median_filter(data, 300)
    
    return model - data_filt
    #return -np.nansum(tell_interp*data)
    

def update_wv_soln(order_fluxes, rel_v, old_wv_solns, tell_wvs, tell_model, star_wvs, star_model, resp_xs, resp_model, xs=None, plot=False):
    new_wv_solns = []

    beta = rel_v/consts.c.to(u.km/u.s).value
    redshift = np.sqrt((1 + beta)/(1 - beta)) - 1

    if xs is None:
        xs = np.arange(10, 2000)

    for order_index in range(9):

        fit_data = np.copy(order_fluxes[order_index])
        order_response = resp_model[order_index]
        #fit_data /= np.nanpercentile(fit_data, 98)#ndi.median_filter(fit_data, 300)
        #fit_data[np.where(np.abs(fit_data - np.median(fit_data)) > 10)] = np.median(fit_data)
        good = np.where(~np.isnan(fit_data))
        
        polyparams = old_wv_solns[order_index]
        polyparams = [0*1e25, polyparams[-5]*1e20, polyparams[-4]*1e15, polyparams[-3]*1e10, polyparams[-2]*1e5, polyparams[-1]]
        fitparams = polyparams
        
    #     if order_index < 6:
    #         tell_model = np.ones(tell_dat['wvs'].shape)
    #     else:
    #         tell_model = tell_dat['trans']
        
        result = scipy.optimize.leastsq(fit_to_model,  fitparams, 
                                        args=(xs[good], fit_data[good], redshift, tell_wvs, tell_model, star_wvs, star_model, resp_xs, order_response))

        #print(result)
        polyparams = result[0][0:]
        delta_beta = 0 #result[0][0]
        redshift = np.sqrt((1 + beta + delta_beta)/(1 - beta - delta_beta)) - 1
        params = [polyparams[0]/1e25, polyparams[1]/1e20, polyparams[2]/1e15, polyparams[3]/1e10, polyparams[4]/1e5, polyparams[5]]
        #print(params, delta_beta * consts.c.to(u.km/u.s).value)
        wvsoln = np.poly1d(params)
        thiswvs = wvsoln(xs)
        #print(thiswvs)


        
        new_wv_solns.append(params)

        if plot:
            wvs_model = wvsoln(xs)

            template_interp = np.interp(wvs_model, star_wvs*(1+redshift), star_model)
            tell_interp = np.interp(wvs_model, tell_wvs, tell_model)
            response_interp = np.interp(xs, resp_xs, order_response)

            model = template_interp * tell_interp * response_interp
            model *= np.nanpercentile(fit_data, 95)/np.nanpercentile(model, 95)


            plt.figure(figsize=(20,4))
            plt.plot(thiswvs, fit_data - 0.25, 'b-')
            plt.plot(wvs_model, model, 'k-', alpha=0.5)


            plt.xlim(wvsoln([0,2048]))
            #plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.001))
            plt.grid()


            plt.figure(figsize=(20,4))
            plt.plot(thiswvs, ndi.gaussian_filter(fit_data,1), 'b-')
            plt.plot(wvs_model, model, 'k-', alpha=0.5)


            plt.xlim([wvs_model[50], wvs_model[500]])
            #plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.001))
            plt.grid()
            plt.show()

    return new_wv_solns

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
    barycorr = sc.radial_velocity_correction(obstime=Time(mjd, format='mjd', scale='utc'), location=keck)
    print(barycorr.to(u.km/u.s).value )

    rel_v = -barycorr.to(u.km/u.s).value + star_v # postiive is redshfit? 
    
    return rel_v