import os
import multiprocessing as mp
from glob import glob
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from kpicdrp.wavecal import *

def plot_kpic_spectrum(arr,wvs=None,arr_err=None,ax_list=None,linestyle="-",linewidth=2,color=None,label=None):
    if ax_list is None:
        _ax_list = []

    if wvs is None:
        wvs = np.arange(arr.shape[1])

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



if __name__ == "__main__":

    try:
        import mkl

        mkl.set_num_threads(1)
    except:
        pass


    ## Change local directory
    kpicpublicdir = "/scr3/jruffio/data/kpic/public_kpic_data/"

    ## Path relative to the public kpic directory
    filelist_spectra = glob(os.path.join(kpicpublicdir, "20200702_HIP_81497", "*fluxes.fits"))
    filename_oldwvs = os.path.join(kpicpublicdir, "utils", "first_guess_wvs_20200607_HIP_81497.fits")
    filename_phoenix_rvstandard = os.path.join(kpicpublicdir, "utils", "HIP_81497_lte03600-1.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
    filename_phoenix_wvs =os.path.join(kpicpublicdir, "utils", "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
    filename_line_width = glob(os.path.join(kpicpublicdir, "20200702_HIP_81497", "calib", "*_line_width_smooth.fits"))[0]
    filelist_atran = glob(os.path.join(kpicpublicdir, "utils","atran","atran_13599_30_*.dat"))

    out_filename = os.path.join(kpicpublicdir, "20200702_HIP_81497", "calib", "20200702_HIP_81497_wvs.fits")

    target_rv = -55.567 #km/s
    N_nodes_wvs=6 # Number of spline nodes for the wavelength solution
    blaze_chunks=5 # Number of chunks of the "blaze profile" (modeled as a piecewise linear function)
    init_grid_search = False # do a rough grid search of the wavcal before running optimizer
    init_grid_dwv = 1e-4#3e-4 #mircrons, how far to go for the grid search. Caution: It can take quite a while!
    fringing = False
    numthreads = 10
    mypool = mp.Pool(processes=numthreads)
    # mypool = None

    # Read an old wavelength array to be used as the first guess
    hdulist = fits.open(filename_oldwvs)
    old_wvs = hdulist[0].data

    hdulist = fits.open(filename_line_width)
    line_width = hdulist[0].data
    line_width_func_list = linewidth2func(line_width,old_wvs)

    # Read the Phoenix model and wavelength corresponding to the RV standard star.
    with fits.open(filename_phoenix_wvs) as hdulist:
        phoenix_wvs = hdulist[0].data / 1.e4
    crop_phoenix = np.where((phoenix_wvs > 1.8 - (2.6 - 1.8) / 2) * (phoenix_wvs < 2.6 + (2.6 - 1.8) / 2))
    phoenix_wvs = phoenix_wvs[crop_phoenix]
    with fits.open(filename_phoenix_rvstandard) as hdulist:
        phoenix_spec = hdulist[0].data[crop_phoenix]

    # read spectra files
    combined_spec,combined_err,avg_baryrv = stellar_spectra_from_files(filelist_spectra)
    # plot_kpic_spectrum(combined_spec[1,:,:])
    # plt.show()

    new_wvs_arr = np.zeros(combined_spec.shape)

    for fib in range(combined_spec.shape[0]):
        if np.nansum(combined_spec[fib,:,:])==0:
            continue
        star_rv = target_rv-avg_baryrv[fib]

        # broaden and create interpolation function for the Phoenix model
        phoenix_line_widths = np.array(pd.DataFrame(line_width_func_list[fib](phoenix_wvs)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
        print("broaden Phoenix model")
        phoenix_conv = convolve_spectrum_line_width(phoenix_wvs, phoenix_spec, phoenix_line_widths, mypool=mypool)
        ##
        phoenix_func = interp1d(phoenix_wvs, phoenix_conv / np.nanmax(phoenix_conv), bounds_error=False,fill_value=np.nan)

        ## Create a interpolation function for the tellurics model (including water and zenith angle)
        atrangridname = os.path.join(kpicpublicdir, "utils","atran", "atran_grid_f{0}.fits".format(fib))
        if 0:
            # enable this section to broaden and save the atran grid of models.
            # The resulting fits file can directly be used to generate a regular grid interpolator as shown below.
            # the broadening is specific to the line width calibration, so it might vary from epoch to epoch and maybe
            # even between fibers. That being said, it might not make a big difference...
            save_atrangrid(filelist_atran, line_width_func_list[fib], atrangridname,mypool=mypool)
            # exit()
        hdulist = fits.open(atrangridname)
        atran_grid =  hdulist[0].data
        water_unique =  hdulist[1].data
        angle_unique =  hdulist[2].data
        atran_wvs =  hdulist[3].data
        hdulist.close()
        ##
        print("atran grid interpolator")
        atran_interpgrid = RegularGridInterpolator((water_unique,angle_unique),atran_grid,method="linear",bounds_error=False,fill_value=0.0)

        # Derive wavecal for a single fiber
        print("start fitting")
        new_wvs_fib,model,out_paras = fit_wavecal_fib(old_wvs[fib,:,:],combined_spec[fib,:,:],combined_err[fib,:,:],
                                                     phoenix_func,star_rv,atran_wvs,atran_interpgrid,
                                                     N_nodes_wvs=N_nodes_wvs,
                                                     blaze_chunks=blaze_chunks,
                                                     init_grid_search = init_grid_search,
                                                     init_grid_dwv = init_grid_dwv,
                                                     fringing=fringing,
                                                     mypool=mypool)
        new_wvs_arr[fib,:,:] = new_wvs_fib

        plt.figure(fib+1,figsize=(12,12))
        ax_list = plot_kpic_spectrum(combined_spec[fib,:,:],wvs=new_wvs_fib,arr_err=combined_err[fib,:,:],color="blue",label="data")
        ax_list = plot_kpic_spectrum(model,wvs=new_wvs_fib,color="orange",label="model",ax_list=ax_list)
        plt.legend()
        print("Saving " + out_filename.replace(".fits","_f{0}.png".format(fib)))
        plt.savefig(out_filename.replace(".fits","_f{0}.png".format(fib)))

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=new_wvs_arr,header=fits.open(filelist_spectra[0])[0].header))
    print("Saving "+ out_filename)
    try:
        hdulist.writeto(out_filename, overwrite=True)
    except TypeError:
        hdulist.writeto(out_filename, clobber=True)
    hdulist.close()

    plt.show()

    # if 0: # fringing test
    #     wvs00 = old_wvs[2,4,:]
    #     spec = combined_spec[2,4,:]
    #     spec = edges2nans(spec)
    #     spec / np.nanmax(spec)
    #     from jbdrp.fit_single_object import LPFvsHPF
    #
    #     spec_lpf = LPFvsHPF(spec,10)[0]
    #
    #     F_vec,G_vec = np.linspace(0.04,0.10,1000),np.linspace(1.00e4,1.2e4,1000)
    #     chi2 = np.zeros((np.size(F_vec),np.size(G_vec)))
    #     for Fid,F in enumerate(F_vec):
    #         for Gid,G in enumerate(G_vec):
    #             delta = (2*np.pi)/wvs00*G
    #             m3 = 1/(1+F*np.sin(delta)**2)*spec_lpf
    #             m3  = m3*np.nansum(spec*m3)/np.nansum(m3**2)
    #             chi2[Fid,Gid] = np.nansum((spec-m3)**2)
    #     myargmin = np.unravel_index(np.argmin(chi2),chi2.shape)
    #     print(myargmin)
    #     print(F_vec[myargmin[0]],G_vec[myargmin[1]])
    #     # (439, 426)
    #     # 0.06636636636636636 10852.852852852853
    #     plt.figure(1)
    #     plt.imshow(chi2,origin="lower")
    #     # plt.imshow(chi2,origin="lower",extent=[G_vec[0],G_vec[-1],F_vec[0],F_vec[-1]],aspect=(G_vec[-1]-G_vec[0])/(F_vec[-1]-F_vec[0]))
    #     plt.figure(2)
    #     delta = (2*np.pi)/wvs00*G_vec[myargmin[1]]
    #     m3 = 1/(1+F_vec[myargmin[0]]*np.sin(delta)**2)*spec_lpf
    #     m3  = m3*np.nansum(spec*m3)/np.nansum(m3**2)
    #     plt.plot(m3)
    #     plt.plot(spec)
    #     plt.plot(spec-m3)
    #     plt.show()