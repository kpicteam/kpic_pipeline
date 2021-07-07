import astropy.io.fits as pyfits
import astropy.time as time
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import median_filter
from astropy.stats import mad_std
from scipy.signal import correlate2d
from copy import copy
import multiprocessing as mp
from scipy.optimize import minimize
import itertools
# from utils.badpix import *
# from utils.misc import *


from scipy.signal import find_peaks
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize

import kpicdrp.data as data

def profile_model(paras,y):
    A, w, y0, B= paras
    return A/np.sqrt(2*np.pi*w**2)*np.exp(-1./(2.*w**2)*(y-y0)**2)+B

def fit_trace_nloglike(paras,datacol,y):
    A, w, y0, rn = paras
    N_d = np.size(np.where(np.isfinite(datacol))[0])
    nloglike = np.nansum((datacol-profile_model([A, w, y0, 0],y))**2/rn**2) + \
               N_d*np.log10(2*np.pi*rn**2)
    return 1/2.*nloglike

def fit_trace_nloglike_background(paras,datacol,y):
    A, w, y0, rn,B = paras
    N_d = np.size(np.where(np.isfinite(datacol))[0])
    nloglike = np.nansum((datacol-profile_model([A, w, y0, B],y))**2/rn**2) + \
               N_d*np.log10(2*np.pi*rn**2)
    return 1/2.*nloglike

def _fit_trace(paras):
    xindices,yindices,data,badpix,fitbackground = paras
    nystamp,nxstamp = data.shape
    # paras0 = [A, w, y0, B, rn] or [A, w, y0, B, rn, g]?
    out = np.zeros((nxstamp,5))
    residuals = np.zeros(data.shape)

    for k in range(nxstamp):
        # print(k)
        badpixcol =  badpix[:,k]
        datacol = data[:,k]*badpixcol
        N_d = np.size(np.where(np.isfinite(datacol))[0])
        skip = False
        if N_d != 0:
            A= np.nansum(datacol)
            w = 2
            y0 = yindices[np.nanargmax(datacol)]
            B = 0
            rn = np.nanstd(datacol)
            if np.abs(rn<1e-14): ### Added by LF 04 May 21 to avoid returning nonsense values
                skip=True
            # g=0
            if fitbackground:
                paras0 = [A, w, y0, rn, B]
            else:
                paras0 = [A, w, y0, rn]
            # print(fit_trace_nloglike(paras0,datacol,yindices)/np.size(datacol))
            # exit()
        else:
            skip = True

        if not skip:
            # tmp_datacol = datacol[np.max([y0-yindices[0]-5,0]):np.min([y0-yindices[0]+5,np.size(datacol)])]
            # skip = np.size(np.where(np.isfinite(tmp_datacol))[0]) < 0.7*np.size(tmp_datacol) or  np.size(np.where(np.isfinite(datacol))[0]) < 0.7*np.size(datacol)
            skip = np.size(np.where(np.isfinite(datacol))[0]) < 0.7 * np.size(datacol)

        if not skip:
            if fitbackground:
                res = minimize(lambda paras:fit_trace_nloglike_background(paras,datacol,yindices,),paras0,method="nelder-mead",options={"xatol":1e-3,"maxiter":1e4})
                out[k,:] = [res.x[0],res.x[1],res.x[2],res.x[4],res.x[3]]
                residuals[:,k] = (datacol - profile_model([res.x[0],res.x[1],res.x[2],res.x[4]],yindices))/res.x[3]
            else:
                res = minimize(lambda paras:fit_trace_nloglike(paras,datacol,yindices,),paras0,method="nelder-mead",options={"xatol":1e-3,"maxiter":1e4})
                out[k,:] = [res.x[0],res.x[1],res.x[2],0,res.x[3]]
                residuals[:,k] = (datacol - profile_model([res.x[0],res.x[1],res.x[2],0],yindices))/res.x[3]

            # plt.plot(yindices,datacol)
            # plt.plot(yindices,profile_model([res.x[0],res.x[1],res.x[2],0],yindices))
            # plt.show()
            # print(res.x[2]<yindices[0] or yindices[-1]<res.x[2])
            # print(res.x[2],yindices[0] , yindices[-1],res.x[2])
            if np.abs(res.x[2]- y0) > 3:
                skip = True
            # print(res.x[1]<0.5 or 3<res.x[1])
            # print(res.x[1],0.5 , 3,res.x[1])
            if res.x[1]<0.5 or 3<res.x[1]:
                skip = True
            # print( res.x[0]<0.0 or 2*np.nanmax(datacol)<res.x[0]/np.sqrt(2*np.pi*res.x[1]**2))
            # print( res.x[0],0.0 , 2*np.nanmax(datacol),res.x[0]/np.sqrt(2*np.pi*res.x[1]**2))
            if res.x[0]<0.0 or 2*np.nanmax(datacol)<res.x[0]/np.sqrt(2*np.pi*res.x[1]**2):
                skip = True

            # print(res.x)
            # print(skip)
            # plt.subplot(1,2,1)
            # plt.plot(datacol,label="data")
            # # plt.plot(profile_model(paras0[0:4],yindices),label="m0")
            # if fitbackground:
            #     plt.plot( profile_model([res.x[0],res.x[1],res.x[2],res.x[4]],yindices),label="m")
            # else:
            #     plt.plot( profile_model([res.x[0],res.x[1],res.x[2],0],yindices),label="m")
            # plt.legend()
            # plt.subplot(1,2,2)
            # plt.plot(res.x[3]+0*datacol,label="sig")
            # plt.plot(residuals[:,k]*res.x[3],label="res")
            # plt.legend()
            # plt.show()

        if skip:
            residuals[:, k]=np.nan
            out[k,:]=np.nan


    return out,residuals

# ----

def tophat(x, hat_left, hat_right):
    if hat_right<hat_left:
        return np.zeros(np.size(x))+np.inf
    else:
        return np.where((hat_left < x) & (x < hat_right), 1, 0)

def objective(params, x, y):
    return np.sum(np.abs(tophat(x, *params) - y))

def fibers_guess(fiber_dataset, N_order=9):
    """
    Find first guess position of the traces from a list of images.

    Args:
        fiber_dataset (data.Dataset): DetectorFrame images for trace calibration
        N_order: Number of orders in the image

    Returns:
        guess_params (data.TraceParams): trace parameters with rough locations and widths fixed to 3 pixels FWHM
    """

    corr = np.zeros((len(fiber_dataset), len(fiber_dataset)))
    im_norm_list = [np.sqrt(np.nansum(frame.data**2)) for frame in fiber_dataset]
    for k, frame1 in enumerate(fiber_dataset):
        for l, frame2 in enumerate(fiber_dataset):
            corr[k,l] = np.nansum(frame1.data * frame2.data)/(im_norm_list[k] * im_norm_list[l])
    sorted_files = []
    while np.sum(corr) != 0:
        ids = np.where(corr[:,np.where(np.nansum(corr,axis=0)!=0)[0][0]] > 0.9)[0]
        sorted_files.append(ids)
        corr[ids,:] = 0
        corr[:,ids] = 0

    fibers_unsorted = {}
    for k, file_ids in enumerate(sorted_files):
        med_frame = np.nanmedian(fiber_dataset[file_ids].data, axis=0)
        background_cutoff = np.percentile(med_frame[np.where(np.isfinite(med_frame))], 97)
        med_frame[np.where(med_frame < background_cutoff)] = 0
        flattened = np.nanmean(med_frame, axis=1)
        peaks = find_peaks(flattened,distance=120,width=[2,None])[0]#,width=[2,None],plateau_size=None
        peaks_val = np.array([flattened[peak] for peak in peaks])
        peaks = peaks[np.argsort(peaks_val)[::-1][0:N_order]]
        peaks = np.sort(peaks)
        x = np.arange(np.size(flattened))
        w = 100
        fiber_guess = []
        for peak in peaks:
            guess = np.array([peak-15,peak+15])
            left,right = np.nanmax([0,peak-w]),np.nanmin([peak+w,np.size(flattened)-1])
            simplex_init_steps = [10,10]
            initial_simplex = np.concatenate([guess[None,:],guess[None,:] + np.diag(simplex_init_steps)],axis=0)
            res = minimize(objective, guess, args=(x[left:right], 2*flattened[left:right]/flattened[peak]), method='Nelder-Mead',
                   options={"xatol": 1e-6, "maxiter": 1e5,"initial_simplex":initial_simplex,"disp":False})
            # fiber_guess.append([int(np.round(res.x[0])),int(np.round(res.x[1]))])
            fiber_guess.append(np.round(res.x).astype(np.int))
            # print(res.x,np.round(res.x).astype(np.int))
            # fiber_guess.append(res.x)
            # plt.plot(x[left:right], tophat(x[left:right], *(res.x))*flattened[peak])
            # plt.plot(x[left:right],flattened[left:right])
        fibers_unsorted[k] = fiber_guess
    # plt.show()
    sorted_fib = np.argsort([np.nanmean(myvals) for myvals in fibers_unsorted.values()])[::-1]


    guess_locs = []
    guess_labels = []
    for k,argfib in enumerate(sorted_fib):
        guess_locs_thisfib = []
        for ends in fibers_unsorted[argfib]:
            num_channels = fiber_dataset[0].data.shape[-1]
            this_order_guesspos = np.interp(np.arange(num_channels), [0, num_channels], ends)
            guess_locs_thisfib.append(this_order_guesspos)
        guess_locs.append(guess_locs_thisfib)
        guess_labels.append("s{0}".format(k + 1))

    guess_locs = np.array(guess_locs)
    guess_labels = np.array(guess_labels)
    guess_widths = np.ones(guess_locs.shape) * (3 / (2 * np.sqrt(2 * np.log(2))))
    guess_params = data.TraceParams(locs=guess_locs, widths=guess_widths, labels=guess_labels, header=fiber_dataset[0].header)
    guess_params.add_parent_filenames(fiber_dataset)
    tnow = time.Time.now()
    guess_params.header['HISTORY'] = "[{0}] Guessed {1} fiber traces".format(str(tnow), len(sorted_fib))

    return guess_params

# ----

def trace(vec,x):
    # vec = trace_calib[fiber_num,order_id, :, para_id]
    vec_cp = copy(vec)

    # vec_lpf = np.array(pd.DataFrame(vec_cp).interpolate(method="linear"))[:,0] #
    # # .fillna(method="bfill").fillna(method="ffill"))[:,0]
    # wherenan_vec_lpf = np.where(np.isnan(vec_lpf))
    # vec_lpf = np.array(pd.DataFrame(vec_lpf).rolling(window=301, center=True).median().fillna(method="bfill").fillna(method="ffill"))[:, 0]
    # vec_hpf = vec - vec_lpf
    # # plt.plot(vec_hpf)
    # # plt.plot(vec_lpf)
    # # plt.show()

    wherefinitevec = np.where(np.isfinite(vec))
    vec_polyfit = np.polyval(np.polyfit(x[wherefinitevec], vec[wherefinitevec], 5), x)
    vec_hpf = vec - vec_polyfit

    vec_hpf_std = mad_std(vec_hpf[np.where(np.isfinite(vec_hpf))])
    # print(vec_hpf[np.where(np.isfinite(vec_hpf))])
    # print(vec_hpf_std)
    where_bad = np.where(np.abs(vec_hpf) > 10 * vec_hpf_std)
    vec[where_bad] = np.nan

    # plt.plot(vec)
    # plt.plot(np.array(pd.DataFrame(vec_cp).rolling(window=301,center=True).median()))
    # plt.plot(np.array(pd.DataFrame(vec_cp).rolling(window=301,center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")),"--")
    # plt.show()

    # vec_lpf[np.where(np.isnan(vec))] = np.nan
    wherefinitevec = np.where(np.isfinite(vec))
    if len(wherefinitevec[0]) == 0:
        # continue
        return None,None

    # vec[where_bad] = [where_bad]
    # print(vec[0:10])
    # plt.plot(vec_cp)
    # plt.plot(vec)
    # plt.show()

    # x_knots = np.concatenate([[0],np.arange(400, 1701, 100).astype(np.int),[trace_calib.shape[2]-1]])  # np.array([wvs_stamp[wvid] for wvid in )
    x_knots = np.linspace(0, len(x)-1, 10,endpoint=True).astype(np.int)  # Changed by LF 05 May 21 to deal with different array sizes
    paras0 = np.array(vec_polyfit[x_knots].tolist())
    simplex_init_steps = np.ones(np.size(paras0))*vec_hpf_std*100
    initial_simplex = np.concatenate([paras0[None, :], paras0[None, :] + np.diag(simplex_init_steps)],axis=0)
    res = minimize(lambda paras: np.nansum((InterpolatedUnivariateSpline(x_knots, paras, k=3, ext=0)(x)-vec)**2),
                   paras0,method="nelder-mead",
                   options={"xatol": 1e-8, "maxiter": 1e5, "initial_simplex": initial_simplex,
                            "disp": False})
    print('paras0',paras0)
    print('res.x',res.x)
    print('vec_hpf_std',vec_hpf_std)
    spl = InterpolatedUnivariateSpline(x_knots, res.x, k=3, ext=0)


    polyfit_trace_calib = vec_polyfit

    # vec_lpf[wherenan_vec_lpf] = polyfit_trace_calib[order_id,wherenan_vec_lpf[0],para_id]
    smooth_trace_calib = spl(x)

    return polyfit_trace_calib,smooth_trace_calib

def smooth(trace_calib):
    print('trace_calib',trace_calib)
    polyfit_trace_calib = trace_calib.copy() 
    smooth_trace_calib = trace_calib.copy()
    # paras0 = [A, w, y0, B, rn] or [A, w, y0, B, rn, g]?
    x = np.arange(0, trace_calib.data.shape[2])
    for fiber_num in np.arange(trace_calib.data.shape[0]):
        for order_id in range(trace_calib.data.shape[1]):

            vec = trace_calib.locs[fiber_num, order_id]
            poly, smoothed = trace(vec,x)
            polyfit_trace_calib.locs[fiber_num, order_id] = poly
            smooth_trace_calib.locs[fiber_num, order_id] = smoothed

            vec = trace_calib.widths[fiber_num, order_id]
            poly, smoothed = trace(vec,x)
            polyfit_trace_calib.widths[fiber_num, order_id] = poly
            smooth_trace_calib.widths[fiber_num, order_id] = smoothed

    tnow = time.Time.now()
    trace_calib.header['HISTORY'] = "[{0}] Fiber params smoothed".format(str(tnow))


    return polyfit_trace_calib,smooth_trace_calib

def guess_star_fiber(image, fiber_params):
    # fibers is a dict wih 0, 1, 2, 3... and for each x1 and x2
    # fiber1=None
    # fiber2=None
    # fiber3=None
    # fiber4=None

    # dictionary... length of... num of fibes

    fiber_templates = []
    fiber_label = fiber_params.labels
    for i, f_num in enumerate(fiber_label): # in order...?
        fiber_template = np.zeros(2048)
        for locs in fiber_params.locs[i]:
            x1 = int(np.min(locs))
            x2 = int(np.max(locs))
            fiber_template[x1:x2] = 1
        fiber_templates.append(fiber_template)
    flattened = np.nanmean(image,axis=1)

    # fiber1_template = np.zeros(2048)
    # for x1,x2 in fiber1:
    #     fiber1_template[x1+10:x2-10] = 1
    # fiber2_template = np.zeros(2048)
    # for x1,x2 in fiber2:
    #     fiber2_template[x1+10:x2-10] = 1
    # fiber3_template = np.zeros(2048)
    # for x1,x2 in fiber3:
    #     fiber3_template[x1+10:x2-10] = 1
    # fiber4_template = np.zeros(2048)
    # for x1,x2 in fiber4:
    #     fiber4_template[x1+10:x2-10] = 1
    # flattened = np.nanmean(image,axis=1)

    # import matplotlib.pyplot as plt
    # plt.plot(flattened/np.nanmax(flattened),label="flat")
    # # plt.plot(fiber1_template,label="0")
    # plt.plot(fiber2_template,label="1")
    # # plt.plot(fiber3_template,label="2")
    # # plt.plot(fiber4_template,label="3")
    # plt.show()

    flat_ftemps = []
    for template in fiber_templates:
        flat_ftemps.append(np.nansum(template * flattened))

    # return np.argmax([np.nansum(fiber1_template * flattened),np.nansum(fiber2_template * flattened),np.nansum(fiber3_template * flattened),np.nansum(fiber4_template * flattened)])
    return fiber_label[np.argmax(flat_ftemps)]


def load_filelist(filelist,background_med_filename,persisbadpixmap_filename):
    hdulist = pyfits.open(background_med_filename)
    background = hdulist[0].data
    background_header = hdulist[0].header
    tint = int(background_header["ITIME"])

    hdulist = pyfits.open(persisbadpixmap_filename)
    persisbadpixmap = hdulist[0].data
    persisbadpixmap_header = hdulist[0].header
    ny,nx = persisbadpixmap.shape

    im_list = []
    badpixmap_list = []
    # fiber_list = []
    header_list = []

    for filename in filelist:
        print(filename)
        hdulist = pyfits.open(filename)
        im = hdulist[0].data.T[:,::-1]
        header = hdulist[0].header
        header_list.append(header)
        if tint != int(header["ITIME"]):
            raise Exception("bad tint {0}, should be {1}: ".format(int(header["ITIME"]),tint) + filename)
        hdulist.close()

        im_skysub = im-background
        badpixmap = persisbadpixmap#*get_badpixmap_from_laplacian(im_skysub,bad_pixel_fraction=1e-2)

        # plt.imshow(im_skysub*badpixmap,interpolation="nearest",origin="lower")
        # plt.show()

        im_list.append(im_skysub)
        badpixmap_list.append(badpixmap)
        # fiber_list.append(guess_star_fiber(im_skysub*badpixmap,fibers))
    cube = np.array(im_list)
    badpixcube = np.array(badpixmap_list)
    # fiber_list = np.array(fiber_list)
    # print(fiber_list)

    # return cube,badpixcube,fiber_list,ny,nx
    return cube,badpixcube,ny,nx

def fit_trace(fiber_dataset, guess_params, fiber_list, numthreads=30, fitbackground=False, return_residuals=False):

    ##calculate traces, FWHM, stellar spec for each fibers
    # fiber,order,x,[y,yerr,FWHM,FHWMerr,flux,fluxerr],
    num_fibers = len(guess_params.labels)

    fiber_list = np.array(fiber_list)

    badpixcube = np.ones(fiber_dataset.data.shape)
    badpixcube[np.isnan(fiber_dataset.data)] == np.nan

    Norders = guess_params.locs.shape[1]
    Nchannels = guess_params.locs.shape[2]
    Ny = fiber_dataset.data.shape[-2]


    trace_calib = np.zeros((num_fibers, Norders, Nchannels, 5))
    residuals = np.zeros((num_fibers, Ny, Nchannels))
    for fiber_num, fiber_label in enumerate(guess_params.labels):
        print('fiber_num,fiber_list',fiber_num,fiber_list)
        print('np.where(fiber_num==fiber_list)',np.where(fiber_num==fiber_list))

        if np.size(np.where(fiber_label == fiber_list)[0]) == 0:
            print('DID CONTINUE')
            continue

        fib_cube = fiber_dataset.data[np.where(fiber_label == fiber_list)[0], :, :]
        fib_badpixcube = badpixcube[np.where(fiber_label == fiber_list)[0], :, :]

        im = np.median(fib_cube*fib_badpixcube,axis=0)
        badpix = np.ones(fib_badpixcube.shape[1::])
        badpix[np.where(np.nansum(fib_badpixcube,axis=0)<np.min([3,fib_badpixcube.shape[0]]))] = np.nan


        pool = mp.Pool(processes=numthreads)

        for order_id, order_locs in enumerate(guess_params.locs[fiber_num]):
            y1 = int(order_locs[0])
            y2 = int(order_locs[-1])
            print('order_id,(y1,y2)',order_id,(y1,y2))

            _y1,_y2 = np.clip(y1-10,0,2047),np.clip(y2+10,0,2047)
            yindices = np.arange(_y1,_y2)

            if 0:
                if order_id != 0 or fiber_num != 3:
                    continue
                xindices = np.arange(1500, 2000)
                out, residuals = _fit_trace((xindices, yindices,
                                             im[_y1:_y2, xindices[0]:xindices[-1] + 1],
                                             badpix[_y1:_y2, xindices[0]:xindices[-1] + 1],
                                             fitbackground))
                plt.subplot(1, 2, 1)
                plt.imshow(im[_y1:_y2, xindices[0]:xindices[-1] + 1])
                plt.subplot(1, 2, 2)
                plt.imshow(residuals)
                print(out)
                plt.colorbar()
                plt.show()
                # print(out,residuals)
                exit()
            else:
                chunk_size=10
                N_chunks = Nchannels//chunk_size
                indices_chunks = []
                data_chunks = []
                badpix_chunks = []
                for k in range(N_chunks-1):
                    indices_chunks.append(np.arange(k*chunk_size,(k+1)*chunk_size))
                    data_chunks.append(im[_y1:_y2,k*chunk_size:(k+1)*chunk_size])
                    badpix_chunks.append(badpix[_y1:_y2,k*chunk_size:(k+1)*chunk_size])
                indices_chunks.append(np.arange((N_chunks-1)*chunk_size,Nchannels))
                data_chunks.append(im[_y1:_y2,(N_chunks-1)*chunk_size:Nchannels])
                badpix_chunks.append(badpix[_y1:_y2,(N_chunks-1)*chunk_size:Nchannels])
                outputs_list = pool.map(_fit_trace, zip(indices_chunks,
                                                            itertools.repeat(yindices),
                                                            data_chunks,
                                                            badpix_chunks,
                                                            itertools.repeat(fitbackground)))

                normalized_psfs_func_list = []
                chunks_ids = []
                for xindices,out in zip(indices_chunks,outputs_list):
                    # print(out)
                    trace_calib[fiber_num,order_id,xindices[0]:xindices[-1]+1,:] = out[0]
                    residuals[fiber_num,_y1:_y2,xindices[0]:xindices[-1]+1] = out[1]
        # print('trace_calib[fiber_num]',trace_calib[fiber_num])
        # print('residuals[fiber_num]',residuals[fiber_num])

        pool.close()
        pool.join()
    
    # The dimensions of trace calib are (4 fibers, 9 orders, 2048 pixels, 5) #[A, w, y0, rn, B]
    # trace_calib[:,:,:,0]: amplitude of the 1D gaussian
    # trace_calib[:,:,:,1]: trace width (1D gaussian sigma)
    # trace_calib[:,:,:,2]: trace y-position
    # trace_calib[:,:,:,3]: noise (ignore)
    # trace_calib[:,:,:,4]: background (ignore)

    trace_params = data.TraceParams(locs=trace_calib[:,:,:,2], widths=trace_calib[:,:,:,1], labels=guess_params.labels, header=fiber_dataset[0].header)
    trace_params.filedir = fiber_dataset[0].filedir
    trace_params.filename = fiber_dataset[0].filename[:-5] + "_trace.fits"
    # add data reduction history
    trace_params.add_parent_filenames(fiber_dataset)
    tnow = time.Time.now()
    trace_params.header['HISTORY'] = "[{0}] Fit {1} fiber traces in {2} orders".format(str(tnow), num_fibers, Norders)

    if return_residuals:
        return trace_params, residuals
    else:
        return trace_params


def add_background_traces(trace_dat):
    """
    Adds fictitious trace location sampling the slit background and detector background (i.e., out of slit "dark")
    Labels for slit background is 'b1, b2, ..'
    Labels for detector background (dark) is 'd1, d2..'

    Args:
        trace_dat (TraceParams): trace locations for the science fibers

    Return:
        new_trace_dat (TraceParams): trace locations with slit background and detector background traces included
    """
    new_trace_dat = trace_dat.copy()

    trace_loc_slit = np.zeros((new_trace_dat.locs.shape[0], new_trace_dat.locs.shape[1], new_trace_dat.locs.shape[2]))
    trace_loc_dark = np.zeros((new_trace_dat.locs.shape[0], new_trace_dat.locs.shape[1], new_trace_dat.locs.shape[2]))

    for order_id in range(new_trace_dat.locs.shape[1]):
        dy1 = np.nanmean(new_trace_dat.locs[0, order_id, :] - new_trace_dat.locs[1, order_id, :]) / 2
        dy2 = np.nanmean(new_trace_dat.locs[0, order_id, :] - new_trace_dat.locs[-1, order_id, :])
        # exit()
        if np.isnan(dy1):
            dy1 = 10
        if np.isnan(dy2):
            dy2 = 40
        for fib_id in range(new_trace_dat.locs.shape[0]):

            trace_loc_slit[fib_id, order_id, :] = new_trace_dat.locs[fib_id, order_id, :] - dy1

            if order_id == 0:
                trace_loc_dark[fib_id, order_id, :] = new_trace_dat.locs[fib_id, order_id, :] + 1.5 * dy2 - fib_id * dy1
            else:
                trace_loc_dark[fib_id, order_id, :] = new_trace_dat.locs[fib_id, order_id, :] - 3 * dy2 + (3 + fib_id) * dy1


    # add slit and dark traces to trace params
    # first the slit backgrounds
    new_trace_dat.locs = np.append(new_trace_dat.ocs, trace_loc_slit, axis=0)
    new_trace_dat.widths = np.append(new_trace_dat.widths, new_trace_dat.widths, axis=0)
    new_trace_dat.labels = new_trace_dat.labels + ['b{0}' for i in range(trace_loc_slit.shape[0])]
    # next the traces
    new_trace_dat.locs = np.append(new_trace_dat.ocs, trace_loc_dark, axis=0)
    new_trace_dat.widths = np.append(new_trace_dat.widths, new_trace_dat.widths, axis=0)
    new_trace_dat.labels = new_trace_dat.labels + ['d{0}' for i in range(trace_loc_dark.shape[0])]


    return new_trace_dat