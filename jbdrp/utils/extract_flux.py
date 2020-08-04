import astropy.io.fits as pyfits
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
from scipy.interpolate import interp1d

def profile_model(paras,y):
    A, w, y0, B= paras
    return A/np.sqrt(2*np.pi*w**2)*np.exp(-1./(2.*w**2)*(y-y0)**2)+B

# def fit_trace_nloglike(paras,datacol,y,w,y0):
#     A1,A2,A3, rn = paras
#     N_d = np.size(np.where(np.isfinite(datacol))[0])
#     nloglike = np.nansum((datacol-profile_model([A1, w[0], y0[0], 0],y)-profile_model([A2, w[1], y0[1], 0],y)-profile_model([A3, w[2], y0[2], 0],y))**2/rn**2) + \
#                N_d*np.log10(2*np.pi*rn**2)
#     return 1/2.*nloglike

def _extract_flux(paras):
    xindices,yindices,data,badpix,line_width,ycenter,fitbackground,badpix_threshold = paras
    nystamp,nxstamp = data.shape
    # paras0 = [A, w, y0, B, rn] or [A, w, y0, B, rn, g]?
    Nfib = line_width.shape[0]
    out = np.zeros((nxstamp,3*Nfib))
    residuals = np.zeros(data.shape)

    # plt.imshow(data,label="data")
    # plt.show()

    for k in range(nxstamp):
        # print(k)
        badpixcol =  badpix[:,k]
        datacol = data[:,k]*badpixcol
        N_d = np.size(np.where(np.isfinite(datacol))[0])
        A = np.zeros(Nfib)
        A_sig = np.zeros(Nfib)
        maxres = np.zeros(Nfib)
        cp_datacol = np.tile(copy(datacol)[None,:],(Nfib+1,1))
        if N_d != 0:
            for fib in range(Nfib):
                if np.isfinite(line_width[fib,k]) and np.isfinite(ycenter[fib,k]):
                    y0int = int(np.round(ycenter[fib,k]-yindices[0]))
                    for fib2 in range(Nfib+1):
                        if fib2 != fib:
                            cp_datacol[fib2,np.max([y0int - 5, 0]):np.min([y0int + 5,np.size(datacol)])] = np.nan
            if fitbackground:
                background = np.nanmedian(cp_datacol[-1])
            else:
                background=0
            for fib in range(Nfib):
                if np.isfinite(line_width[fib,k]) and np.isfinite(ycenter[fib,k]):
                    y0int = int(np.round(ycenter[fib,k]-yindices[0]))
                    tmp_datacol = datacol[np.max([y0int - 5, 0]):np.min([y0int + 5,np.size(datacol)])]-background
                    m1 = profile_model([1,line_width[fib,k],ycenter[fib,k],0],yindices)[np.max([y0int - 5, 0]):np.min([y0int + 5,np.size(datacol)])]
                    where_finite = np.where(np.isfinite(tmp_datacol))
                    # print(np.size(where_finite[0]) )
                    if np.size(where_finite[0]) >= 7:
                        deno = np.nansum(m1[where_finite]**2)
                        A[fib]= np.nansum(tmp_datacol[where_finite]*m1[where_finite])/deno
                        A_sig[fib] = 1/np.sqrt(deno)
                        maxres[fib] = np.nanmax(tmp_datacol-A[fib]*m1)
                    else:
                        A[fib] = np.nan
                        A_sig[fib] = np.nan
                        maxres[fib] = np.nan
                else:
                    A[fib] = np.nan
                    A_sig[fib] = np.nan
                    maxres[fib] = np.nan

            # plt.plot(cp_datacol[0,:])
            # plt.plot(cp_datacol[1,:])
            # plt.plot(cp_datacol[2,:])
            # plt.plot(cp_datacol[3,:])
            # plt.show()

            cp_datacol_std = np.nanstd(cp_datacol,axis=1)
            # print(cp_datacol_std)
            rn = cp_datacol_std[Nfib]
            A_sig = [s1*s2 for s1,s2 in zip(A_sig,cp_datacol_std[0:Nfib])]

            residuals[:,k] = datacol-background
            for fib in range(Nfib):
                if np.isfinite(A[fib]) and np.isfinite(A_sig[fib]):
                    residuals[:, k] -= profile_model([A[fib],line_width[fib,k],ycenter[fib,k],0],yindices)

            out[k, :] = np.concatenate([A,A_sig,np.abs((maxres) / mad_std(residuals[:, k][np.where(np.isfinite(residuals[:, k]))]))],axis=0)

            # out[k, 2*Nfib] = np.abs((np.nanmax(residuals[:, k]) - np.nanmedian(cp_datacol[3, :])) / mad_std(residuals[:, k][np.where(np.isfinite(residuals[:, k]))]))
            # if np.abs((np.nanmax(residuals[:, k])-np.nanmedian(cp_datacol[3,:])) / mad_std(residuals[:, k][np.where(np.isfinite(residuals[:, k]))]))>badpix_threshold:
            #     out[k, :2*Nfib] = np.nan

            # print(A)
            # # print(np.abs((np.nanmax(residuals[:, k])-np.nanmedian(cp_datacol[3,:])) / mad_std(residuals[:, k][np.where(np.isfinite(residuals[:, k]))])))
            # # print(maxres)
            # print(np.abs((maxres) / mad_std(residuals[:, k][np.where(np.isfinite(residuals[:, k]))])))
            # plt.plot(datacol,label="data")
            # plt.plot(residuals[:, k],label="res")
            # plt.plot(cp_datacol[3,:,],label="masked data")
            # plt.legend()
            # plt.show()

            # plt.subplot(1,2,1)
            # print(out[k, :])
            # plt.plot(datacol,label="data",linestyle="--")
            # # plt.plot(profile_model(paras0[0:4],yindices),label="m0")
            # plt.plot(profile_model([A[0],line_width[0,k],ycenter[0,k],0],yindices)+background ,label="m1")
            # plt.plot(profile_model([A[1],line_width[1,k],ycenter[1,k],0],yindices)+background,label="m2")
            # plt.plot(profile_model([A[2],line_width[2,k],ycenter[2,k],0],yindices)+background,label="m3")
            # plt.legend()
            # plt.subplot(1,2,2)
            # plt.plot(rn+0*datacol,label="sig")
            # plt.plot(residuals[:,k]*rn,label="res")
            # plt.plot(background+0*datacol,label="background")
            # print(background)
            # plt.legend()
            # plt.show()
        else:
            residuals[:, k]=np.nan
            out[k,:]=np.nan

    return out,residuals

def extract_flux(image, badpixmap,line_width,trace_loc,numthreads=None,fitbackground=False,bad_pixel_fraction=0.01):
    # fiberall = [[30,150],[220,330],[420,520],[640,720],[865,930],[1090,1170],[1320,1420],[1570,1690],[1840,1980]]
    ny,nx = image.shape
    Nfib = trace_loc.shape[0]
    # for m, p in zip(np.nanmin(trace_loc, axis=(0, 2)), np.nanmax(trace_loc, axis=(0, 2))):
    #     print(m,p)
    badpix_threshold = np.inf

    fiberall = []
    for m, p in zip(np.nanmin(trace_loc, axis=(0, 2)), np.nanmax(trace_loc, axis=(0, 2))):
        if np.isnan(m) or np.isnan(p):
            fiberall.append((None,None))
        else:
            fiberall.append((int(np.floor(m))-10,int(np.ceil(p))+10))

    # print(fiberall)
    fluxes = np.zeros(trace_loc.shape)+np.nan
    errors = np.zeros(trace_loc.shape)+np.nan
    residuals = np.zeros(image.shape)+np.nan
    fitmetric = np.zeros(trace_loc.shape)+np.nan

    if numthreads is not None:
        numthreads=mp.cpu_count()
    pool = mp.Pool(processes=numthreads)

    for order_id,(y1,y2) in enumerate(fiberall):
    # if 1:
    #     order_id, (y1, y2) = 8, fiberall[8]
        print(order_id,(y1,y2))
        if y1 is None or y2 is None:
            continue
        yindices = np.arange(y1,y2)
        # star_stamp = star_im[x1:x2,:]
        # plt.imshow(star_stamp,origin="lower")
        # plt.show()
        # exit()

        # if order_id != 6:
        #     continue

        if 0:
            print(y1,y2)
            xindices = np.arange(1130,1650)
            out,residuals = _extract_flux((xindices,yindices,
                                         image[y1:y2,xindices[0]:xindices[-1]+1],
                                         badpixmap[y1:y2,xindices[0]:xindices[-1]+1],
                                          line_width[:,order_id,xindices[0]:xindices[-1]+1],
                                          trace_loc[:,order_id,xindices[0]:xindices[-1]+1],fitbackground,badpix_threshold))
            # metric = out[:,-1]
            out = out[:,0:6]
            plt.subplot(1,2,1)
            plt.imshow(image[y1:y2,xindices[0]:xindices[-1]+1])
            plt.plot(trace_loc[0,order_id,xindices[0]:xindices[-1]+1],label="0",color="white")
            plt.plot(trace_loc[1,order_id,xindices[0]:xindices[-1]+1],label="1",color="white")
            plt.plot(trace_loc[2,order_id,xindices[0]:xindices[-1]+1],label="2",color="white")
            plt.subplot(1,2,2)
            plt.imshow(residuals)

            plt.figure(2)
            plt.plot(out[:,0],label="0")
            plt.plot(out[:,1],label="1")
            plt.plot(out[:,2],label="2")
            plt.legend()
            plt.show()
            # print(out,residuals)
            exit()
        else:
            chunk_size=10
            N_chunks = nx//chunk_size
            indices_chunks = []
            data_chunks = []
            badpix_chunks = []
            w_chunks = []
            y0_chunks = []
            for k in range(N_chunks-1):
                indices_chunks.append(np.arange(k*chunk_size,(k+1)*chunk_size))
                data_chunks.append(image[y1:y2,k*chunk_size:(k+1)*chunk_size])
                badpix_chunks.append(badpixmap[y1:y2,k*chunk_size:(k+1)*chunk_size])
                w_chunks.append(line_width[:,order_id,k*chunk_size:(k+1)*chunk_size])
                y0_chunks.append(trace_loc[:,order_id,k*chunk_size:(k+1)*chunk_size])
            indices_chunks.append(np.arange((N_chunks-1)*chunk_size,nx))
            data_chunks.append(image[y1:y2,(N_chunks-1)*chunk_size:nx])
            badpix_chunks.append(badpixmap[y1:y2,(N_chunks-1)*chunk_size:nx])
            w_chunks.append(line_width[:,order_id,(N_chunks-1)*chunk_size:nx])
            y0_chunks.append(trace_loc[:,order_id,(N_chunks-1)*chunk_size:nx])
            outputs_list = pool.map(_extract_flux, zip(indices_chunks,
                                                    itertools.repeat(yindices),
                                                    data_chunks,
                                                    badpix_chunks,
                                                    w_chunks,
                                                    y0_chunks,
                                                    itertools.repeat(fitbackground),
                                                    itertools.repeat(badpix_threshold)))

            normalized_psfs_func_list = []
            chunks_ids = []
            for xindices,out in zip(indices_chunks,outputs_list):
                fluxes[:,order_id,xindices[0]:xindices[-1]+1] = out[0][:,0:Nfib].T
                errors[:,order_id,xindices[0]:xindices[-1]+1] = out[0][:,Nfib:2*Nfib].T
                fitmetric[:,order_id,xindices[0]:xindices[-1]+1] = out[0][:,2*Nfib::].T
                # fluxes[order_id,xindices[0]:xindices[-1]+1] = out[0][1]
                # errors[order_id,xindices[0]:xindices[-1]+1] = out[0][4]
                # fluxes[order_id,xindices[0]:xindices[-1]+1] = out[0][2]
                # errors[order_id,xindices[0]:xindices[-1]+1] = out[0][6]
                residuals[y1:y2,xindices[0]:xindices[-1]+1] = out[1]
                # exit()

            # # paras0 = [A, w, y0, B, rn] or [A, w, y0, B, rn, g]?
            # plt.figure(1)
            # plt.subplot(5,1,1)
            # plt.plot(trace_calib[order_id,:,0])
            # plt.ylabel("Flux")
            # plt.subplot(5,1,2)
            # plt.plot(trace_calib[order_id,:,1])
            # plt.ylabel("line width")
            # plt.subplot(5,1,3)
            # plt.plot(trace_calib[order_id,:,2])
            # plt.ylabel("y pos")
            # plt.subplot(5,1,4)
            # plt.plot(trace_calib[order_id,:,3])
            # plt.ylabel("background")
            # plt.subplot(5,1,5)
            # plt.plot(trace_calib[order_id,:,4])
            # plt.ylabel("noise")
            #
            #
            # plt.figure(2)
            # plt.subplot(1, 2, 1)
            # plt.imshow(star_im)
            # plt.clim([0,500])
            # plt.subplot(1, 2, 2)
            # plt.imshow(residuals)
            # print(out)
            # plt.colorbar()
            # plt.show()

    pool.close()
    pool.join()

    for fib in range(fitmetric.shape[0]):
        for order_id in range(fitmetric.shape[1]):
            # plt.plot(fitmetric[fib,order_id,:])
            # plt.show()
            where_finite_metric = np.where(np.isfinite(fitmetric[fib,order_id,:]))
            med_val = np.nanmedian(fitmetric[fib,order_id,:])
            mad_val = mad_std(fitmetric[fib,order_id,:][where_finite_metric])
            hist, bin_edges = np.histogram(fitmetric[fib,order_id,:][where_finite_metric], bins=np.linspace(-100 * mad_val + med_val, 100 * mad_val + med_val, 200 * 10))
            bin_center = (bin_edges[1::] + bin_edges[0:len(bin_edges) - 1]) / 2.
            # ind = np.argsort(hist)
            # cum_posterior = np.zeros(np.shape(hist))
            cum_posterior = np.cumsum(hist)
            cum_posterior = cum_posterior / np.max(cum_posterior)
            # plt.plot(bin_center,hist/np.max(hist))
            # plt.plot(bin_center,cum_posterior)
            # plt.show()
            rf = interp1d(cum_posterior, bin_center, bounds_error=False, fill_value=np.nan)
            upper_bound = rf(1-bad_pixel_fraction)

            where_bad_pixels = np.where((fitmetric[fib,order_id,:] > upper_bound))

            fluxes[fib,order_id,:][where_bad_pixels] = np.nan
            errors[fib,order_id,:][where_bad_pixels] = np.nan


    return fluxes,errors,residuals