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
import itertools
from utils_2020.badpix import *
from utils_2020.misc import *
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize

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

        # print(skip)
        if not skip:
            if fitbackground:
                res = minimize(lambda paras:fit_trace_nloglike_background(paras,datacol,yindices,),paras0,method="nelder-mead",options={"xatol":1e-3,"maxiter":1e4})
                out[k,:] = [res.x[0],res.x[1],res.x[2],res.x[4],res.x[3]]
                residuals[:,k] = (datacol - profile_model([res.x[0],res.x[1],res.x[2],res.x[4]],yindices))/res.x[3]
            else:
                res = minimize(lambda paras:fit_trace_nloglike(paras,datacol,yindices,),paras0,method="nelder-mead",options={"xatol":1e-3,"maxiter":1e4})
                out[k,:] = [res.x[0],res.x[1],res.x[2],0,res.x[3]]
                residuals[:,k] = (datacol - profile_model([res.x[0],res.x[1],res.x[2],0],yindices))/res.x[3]

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


fiber1 = np.array(
    [[70, 150], [260, 330], [460, 520], [680 - 10, 720 + 10], [900 - 15, 930 + 15], [1120 - 5, 1170 + 5],
     [1350, 1420], [1600, 1690], [1870, 1980]]) + 15
fiber2 = np.array(
    [[50, 133], [240, 320], [440, 510], [650, 710], [880 - 15, 910 + 15], [1100 - 5, 1150 + 5], [1330, 1400],
     [1580, 1670], [1850, 1960]]) + 15
fiber3 = np.array(
    [[30, 120], [220, 300], [420, 490], [640 - 5, 690 + 5], [865 - 20, 890 + 20], [1090 - 10, 1130 + 10],
     [1320, 1380], [1570, 1650], [1840, 1940]]) + 10
fiber4 = np.array(
    [[30, 120], [220, 300], [420, 490], [640 - 5, 690 + 5], [865 - 20, 890 + 20], [1090 - 10, 1130 + 10],
     [1320, 1380], [1570, 1650], [1840, 1940]]) - 10
fibers = {0: fiber1, 1: fiber2, 2: fiber3, 3: fiber4}
if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    Nfib = 4
    usershift=0
    fitbackground = False
    mykpicdir = "/scr3/jruffio/data/kpic/"

    # mydir = os.path.join(mykpicdir,"20200607_5_Vul")
    # mydir = os.path.join(mykpicdir,"20200607_ups_Her")
    # mydir = os.path.join(mykpicdir,"20200608_zet_Aql")
    # mydir = os.path.join(mykpicdir,"20200608_d_Sco")
    # mydir = os.path.join(mykpicdir,"20200609_ups_Her")
    # mydir = os.path.join(mykpicdir,"20200609_d_Sco")
    # mydir = os.path.join(mykpicdir,"20200609_kap_And")
    # mydir = os.path.join(mykpicdir,"20200609_zet_Aql")
    # mydir,usershift = os.path.join(mykpicdir,"20200701_ups_Her"),30
    # mydir,usershift = os.path.join(mykpicdir,"20200702_ups_Her"),30
    mydir,usershift = os.path.join(mykpicdir,"20200703_ups_Her"),30



    mydate = os.path.basename(mydir).split("_")[0]

    if 1:
        #background
        background_med_filename = glob(os.path.join(mydir,"calib","*background*.fits"))[0]
        hdulist = pyfits.open(background_med_filename)
        background = hdulist[0].data
        background_header = hdulist[0].header
        tint = int(background_header["ITIME"])
        coadds = int(background_header["COADDS"])

        persisbadpixmap_filename = glob(os.path.join(mydir,"calib","*persistent_badpix*.fits"))[0]
        hdulist = pyfits.open(persisbadpixmap_filename)
        persisbadpixmap = hdulist[0].data
        persisbadpixmap_header = hdulist[0].header
        ny,nx = persisbadpixmap.shape

        filelist = glob(os.path.join(mydir, "raw", "*.fits"))
        filelist.sort()


        im_list = []
        badpixmap_list = []
        fiber_list = []
        header_list = []
        #[1 1 1 1 2 2 2 2 1 2 1 3 1 1 1 1]
        for filename in filelist:
            print(filename)
            hdulist = pyfits.open(filename)
            im = hdulist[0].data.T[:,::-1]
            header = hdulist[0].header
            header_list.append(header)
            if tint != int(header["ITIME"]) or coadds != int(header["COADDS"]):
                raise Exception("bad tint {0} or coadds {1}, should be {2} and {3}: ".format(int(header["ITIME"]),int(header["COADDS"]),tint,coadds) + filename)
            hdulist.close()

            im_skysub = im-background
            badpixmap = persisbadpixmap#*get_badpixmap_from_laplacian(im_skysub,bad_pixel_fraction=1e-2)

            # plt.imshow((im_skysub*badpixmap)[1550:1750,0:100],interpolation="nearest",origin="lower")
            # plt.clim([0,50])
            # plt.show()

            im_list.append(im_skysub)
            badpixmap_list.append(badpixmap)
            fiber_list.append(guess_star_fiber(im_skysub*badpixmap,usershift=usershift,fiber1=fiber1,fiber2=fiber2,fiber3=fiber3,fiber4=fiber4))
        cube = np.array(im_list)
        badpixcube = np.array(badpixmap_list)
        fiber_list = np.array(fiber_list)
        print(fiber_list)
    # exit()

    if 1:
        ##calculate traces, FWHM, stellar spec for each fibers
        # fiber,order,x,[y,yerr,FWHM,FHWMerr,flux,fluxerr],
        trace_calib = np.zeros((Nfib,9,nx,5))
        residuals = np.zeros((Nfib,ny,nx))
        for fiber_num in np.arange(0,Nfib):
            print(np.where(fiber_num==fiber_list))

            if np.size(np.where(fiber_num==fiber_list)[0]) == 0:
                continue

            fib_cube = cube[np.where(fiber_num == fiber_list)[0], :, :]
            fib_badpixcube = badpixcube[np.where(fiber_num == fiber_list)[0], :, :]

            im = np.median(fib_cube*fib_badpixcube,axis=0)
            badpix = np.ones(fib_badpixcube.shape[1::])
            badpix[np.where(np.nansum(fib_badpixcube,axis=0)<np.min([3,fib_badpixcube.shape[0]]))] = np.nan

            numthreads=30
            pool = mp.Pool(processes=numthreads)

            for order_id,(y1,y2) in enumerate(fibers[fiber_num]+usershift):
                print(order_id,(y1,y2))
                yindices = np.arange(y1,y2)

                if 0:
                    if order_id != 5 or fiber_num != 0:
                        continue
                    xindices = np.arange(1500,2000)
                    out,residuals = _fit_trace((xindices,yindices,
                                                 im[y1:y2,xindices[0]:xindices[-1]+1],
                                                 badpix[y1:y2,xindices[0]:xindices[-1]+1],
                                                fitbackground))
                    plt.subplot(1,2,1)
                    plt.imshow(im[y1:y2,xindices[0]:xindices[-1]+1])
                    plt.subplot(1,2,2)
                    plt.imshow(residuals)
                    print(out)
                    plt.colorbar()
                    plt.show()
                    # print(out,residuals)
                    exit()
                else:
                    chunk_size=10
                    N_chunks = nx//chunk_size
                    indices_chunks = []
                    data_chunks = []
                    badpix_chunks = []
                    for k in range(N_chunks-1):
                        indices_chunks.append(np.arange(k*chunk_size,(k+1)*chunk_size))
                        data_chunks.append(im[y1:y2,k*chunk_size:(k+1)*chunk_size])
                        badpix_chunks.append(badpix[y1:y2,k*chunk_size:(k+1)*chunk_size])
                    indices_chunks.append(np.arange((N_chunks-1)*chunk_size,nx))
                    data_chunks.append(im[y1:y2,(N_chunks-1)*chunk_size:nx])
                    badpix_chunks.append(badpix[y1:y2,(N_chunks-1)*chunk_size:nx])
                    outputs_list = pool.map(_fit_trace, zip(indices_chunks,
                                                                itertools.repeat(yindices),
                                                                data_chunks,
                                                                badpix_chunks,
                                                                itertools.repeat(fitbackground)))

                    normalized_psfs_func_list = []
                    chunks_ids = []
                    for xindices,out in zip(indices_chunks,outputs_list):
                        trace_calib[fiber_num,order_id,xindices[0]:xindices[-1]+1,:] = out[0]
                        residuals[fiber_num,y1:y2,xindices[0]:xindices[-1]+1] = out[1]

            pool.close()
            pool.join()


        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=trace_calib,header=header_list[0]))
        out = os.path.join(mydir, "calib", mydate+"_trace_allparas.fits")
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=trace_calib[:,:,:,1],header=header_list[0]))
        out = os.path.join(mydir, "calib", mydate+"_line_width.fits")
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=trace_calib[:,:,:,2],header=header_list[0]))
        out = os.path.join(mydir, "calib", mydate+"_trace_loc.fits")
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=residuals,header=header_list[0]))
        out = os.path.join(mydir, "calib", mydate+"_trace_residuals.fits")
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        # exit()

    if 1:
        out = os.path.join(mydir, "calib", mydate+"_trace_allparas.fits")
        hdulist = pyfits.open(out)
        trace_calib = hdulist[0].data
        header = hdulist[0].header

        if "20191215_kap_And" in mydir:
            trace_calib[:,0:8,:,:] = trace_calib[:,1::,:,:]
            trace_calib[:, 8, :, :] = np.nan

        polyfit_trace_calib = np.zeros(trace_calib.shape)+np.nan
        smooth_trace_calib = np.zeros(trace_calib.shape)+np.nan
        # paras0 = [A, w, y0, B, rn] or [A, w, y0, B, rn, g]?
        x = np.arange(0, trace_calib.shape[2])
        for fiber_num in np.arange(0,Nfib):
            for order_id in np.arange(0,9):
                for para_id in np.arange(trace_calib.shape[3]):
                    print("fiber_num",fiber_num,"order_id",order_id, "para_id", para_id)
                    vec = trace_calib[fiber_num,order_id, :, para_id]
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
                        continue
                    # vec[where_bad] = [where_bad]
                    # print(vec[0:10])
                    # plt.plot(vec_cp)
                    # plt.plot(vec)
                    # plt.show()

                    # x_knots = np.concatenate([[0],np.arange(400, 1701, 100).astype(np.int),[trace_calib.shape[2]-1]])  # np.array([wvs_stamp[wvid] for wvid in )
                    x_knots = np.linspace(0, 2047, 10,endpoint=True).astype(np.int)  # np.array([wvs_stamp[wvid] for wvid in )
                    paras0 = np.array(vec_polyfit[x_knots].tolist())
                    simplex_init_steps = np.ones(np.size(paras0))*vec_hpf_std*100
                    initial_simplex = np.concatenate([paras0[None, :], paras0[None, :] + np.diag(simplex_init_steps)],axis=0)
                    res = minimize(lambda paras: np.nansum((InterpolatedUnivariateSpline(x_knots, paras, k=3, ext=0)(x)-vec)**2),
                                   paras0,method="nelder-mead",
                                   options={"xatol": 1e-8, "maxiter": 1e5, "initial_simplex": initial_simplex,
                                            "disp": False})
                    print(paras0)
                    print(res.x)
                    print(vec_hpf_std)
                    spl = InterpolatedUnivariateSpline(x_knots, res.x, k=3, ext=0)


                    polyfit_trace_calib[fiber_num,order_id, :, para_id] = vec_polyfit#

                    # vec_lpf[wherenan_vec_lpf] = polyfit_trace_calib[order_id,wherenan_vec_lpf[0],para_id]
                    smooth_trace_calib[fiber_num,order_id, :, para_id] = spl(x)

                    # # [A, w, y0, B, rn]
                    if 0:
                        plt.plot(spl(x),color="red")
                        # plt.plot(vec_cp)
                        plt.plot(vec,color="blue")#-np.polyval(np.polyfit(x[np.where(np.isfinite(vec))], vec[np.where(np.isfinite(vec))], 5), x))
                        # plt.plot(np.polyval(np.polyfit(x[np.where(np.isfinite(vec))], vec[np.where(np.isfinite(vec))], 5), x),color="green")
                        plt.scatter(x_knots,res.x)
                        # plt.plot(vec_lpf)#-np.polyval(np.polyfit(x[np.where(np.isfinite(vec))], vec[np.where(np.isfinite(vec))], 5), x))
                        # plt.plot(vec_hpf)
                        # plt.plot(np.ones(vec_hpf.shape)*vec_hpf_std)
                        # plt.plot(np.polyval(np.polyfit(x[np.where(np.isfinite(vec))], vec[np.where(np.isfinite(vec))], 5), x))
                        plt.show()


        # exit()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=smooth_trace_calib[:,:,:,1],header=header))
        hdulist.append(pyfits.ImageHDU(data=trace_calib[:,:,:,1]-smooth_trace_calib[:,:,:,1]))
        out = os.path.join(mydir, "calib", mydate+"_line_width_smooth.fits")
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=smooth_trace_calib[:,:,:,2],header=header))
        hdulist.append(pyfits.ImageHDU(data=trace_calib[:,:,:,2]-smooth_trace_calib[:,:,:,2]))
        out = os.path.join(mydir, "calib", mydate+"_trace_loc_smooth.fits")
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        hdulist.close()

        if 1: #plot
            im = np.nanmean(cube,axis=0)
            trace_loc_filename = glob(os.path.join(mydir, "calib", "*_trace_loc_smooth.fits"))[0]
            hdulist = pyfits.open(trace_loc_filename)
            trace_loc = hdulist[0].data
            trace_loc[np.where(trace_loc == 0)] = np.nan
            print(trace_loc.shape)
            # plt.figure(1)
            # for order_id in range(9):
            #     plt.subplot(9, 1, 9-order_id)
            #     plt.plot(trace_loc[1,order_id,:],linestyle="-",linewidth=2)
            #     plt.legend()
            # plt.show()

            trace_loc_slit = np.zeros((trace_loc.shape[0], trace_loc.shape[1], trace_loc.shape[2]))
            trace_loc_dark = np.zeros((trace_loc.shape[0] * 2, trace_loc.shape[1], trace_loc.shape[2]))
            for order_id in range(9):
                dy1 = np.nanmean(trace_loc[0, order_id, :] - trace_loc[1, order_id, :]) / 2
                dy2 = np.nanmean(trace_loc[0, order_id, :] - trace_loc[3, order_id, :])
                # exit()
                if np.isnan(dy1):
                    dy1 = 10
                if np.isnan(dy2):
                    dy2 = 40
                print(dy1, dy2)

                trace_loc_slit[0, order_id, :] = trace_loc[0, order_id, :] - dy1
                trace_loc_slit[1, order_id, :] = trace_loc[1, order_id, :] - dy1
                trace_loc_slit[2, order_id, :] = trace_loc[2, order_id, :] - dy1
                trace_loc_slit[3, order_id, :] = trace_loc[3, order_id, :] - dy1

                if order_id == 0:
                    trace_loc_dark[0, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 0 * dy1
                    trace_loc_dark[1, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 1 * dy1
                    trace_loc_dark[2, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 2 * dy1
                    trace_loc_dark[3, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 3 * dy1

                    trace_loc_dark[4, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 4 * dy1
                    trace_loc_dark[5, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 5 * dy1
                    trace_loc_dark[6, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 6 * dy1
                    trace_loc_dark[7, order_id, :] = trace_loc[0, order_id, :] + 1.5 * dy2 - 7 * dy1
                else:
                    trace_loc_dark[0, order_id, :] = trace_loc[0, order_id, :] - 3 * dy2 + 3 * dy1
                    trace_loc_dark[1, order_id, :] = trace_loc[1, order_id, :] - 3 * dy2 + 4 * dy1
                    trace_loc_dark[2, order_id, :] = trace_loc[2, order_id, :] - 3 * dy2 + 5 * dy1
                    trace_loc_dark[3, order_id, :] = trace_loc[3, order_id, :] - 3 * dy2 + 6 * dy1

                    trace_loc_dark[4, order_id, :] = trace_loc[0, order_id, :] - 2 * dy2 + 2 * dy1
                    trace_loc_dark[5, order_id, :] = trace_loc[1, order_id, :] - 2 * dy2 + 3 * dy1
                    trace_loc_dark[6, order_id, :] = trace_loc[2, order_id, :] - 2 * dy2 + 4 * dy1
                    trace_loc_dark[7, order_id, :] = trace_loc[3, order_id, :] - 2 * dy2 + 5 * dy1

            plt.figure(1)
            plt.imshow(im,origin="lower")
            plt.clim([np.nanmedian(im),np.nanmedian(im)+50])
            for order_id in range(9):
                for fib in range(trace_loc.shape[0]):
                    plt.plot(trace_loc[fib, order_id, :], label="fibers", color="cyan",linestyle="--",linewidth=1)
                plt.plot(trace_loc[0, order_id, :], label="fibers", color="cyan",linestyle="-",linewidth=2)
                for fib in np.arange(0,trace_loc_slit.shape[0]):
                    plt.plot(trace_loc_slit[fib, order_id, :], label="background", color="red",linestyle="-.",linewidth=1)
                plt.plot(trace_loc_slit[0, order_id, :], label="background", color="red",linestyle="-",linewidth=1)
                for fib in np.arange(0,trace_loc_dark.shape[0]):
                    plt.plot(trace_loc_dark[fib, order_id, :], label="dark", color="white",linestyle=":",linewidth=2)
                plt.plot(trace_loc_dark[0, order_id, :], label="dark", color="white",linestyle="-",linewidth=2)
            plt.xlim([0,im.shape[1]])
            plt.ylim([0,im.shape[0]])
            plt.show()
    exit()