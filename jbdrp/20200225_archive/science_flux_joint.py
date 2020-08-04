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

def profile_model(paras,y):
    A, w, y0, B= paras
    return A/np.sqrt(2*np.pi*w**2)*np.exp(-1./(2.*w**2)*(y-y0)**2)+B

def fit_trace_nloglike(paras,datacol,y,w,y0):
    A1,A2,A3, rn = paras
    N_d = np.size(np.where(np.isfinite(datacol))[0])
    nloglike = np.nansum((datacol-profile_model([A1, w[0], y0[0], 0],y)-profile_model([A2, w[1], y0[1], 0],y)-profile_model([A3, w[2], y0[2], 0],y))**2/rn**2) + \
               N_d*np.log10(2*np.pi*rn**2)
    return 1/2.*nloglike

def _extract_flux(paras):
    xindices,yindices,data,badpix,line_width,ycenter = paras
    nystamp,nxstamp = data.shape
    # paras0 = [A, w, y0, B, rn] or [A, w, y0, B, rn, g]?
    out = np.zeros((nxstamp,6))
    residuals = np.zeros(data.shape)

    for k in range(nxstamp):
        # print(k)
        badpixcol =  badpix[:,k]
        datacol = data[:,k]*badpixcol
        N_d = np.size(np.where(np.isfinite(datacol))[0])
        if N_d != 0:
            cp_datacol = np.tile(copy(datacol)[None,:],(4,1))
            A = np.zeros(3)
            A_sig = np.zeros(3)
            for fib in range(3):
                if np.isfinite(line_width[fib,k]) and np.isfinite(ycenter[fib,k]):
                    # print(ycenter[fib,k])
                    y0int = int(np.round(ycenter[fib,k]-yindices[0]))
                    # plt.plot(datacol)
                    # plt.show()
                    tmp_datacol = datacol[np.max([y0int - 5, 0]):np.min([y0int + 5,np.size(datacol)])]
                    m1 = profile_model([1,line_width[fib,k],ycenter[fib,k],0],yindices)[np.max([y0int - 5, 0]):np.min([y0int + 5,np.size(datacol)])]
                    for fib2 in range(4):
                        if fib2 != fib:
                            cp_datacol[fib2,np.max([y0int - 5, 0]):np.min([y0int + 5,np.size(datacol)])] = np.nan
                    where_finite = np.where(np.isfinite(tmp_datacol))
                    # print(np.size(where_finite[0]) )
                    if np.size(where_finite[0]) >= 7:
                        deno = np.nansum(m1[where_finite]**2)
                        A[fib]= np.nansum(tmp_datacol[where_finite]*m1[where_finite])/deno
                        A_sig[fib] = 1/np.sqrt(deno)
                    else:
                        A[fib] = np.nan
                        A_sig[fib] = np.nan


            cp_datacol_std = np.nanstd(cp_datacol,axis=1)
            rn = cp_datacol_std[3]
            A_sig = [s1*s2 for s1,s2 in zip(A_sig,cp_datacol_std[0:3])]
            out[k, :] = np.concatenate([A,A_sig],axis=0)

            residuals[:,k] = datacol /rn
            for fib in range(3):
                if np.isfinite(A[fib]) and np.isfinite(A_sig[fib]):
                    residuals[:, k] -= profile_model([A[fib],line_width[fib,k],ycenter[fib,k],0],yindices)/rn

            # plt.subplot(1,2,1)
            # print(out[k, :])
            # plt.plot(datacol,label="data")
            # # plt.plot(profile_model(paras0[0:4],yindices),label="m0")
            # plt.plot(profile_model([A[0],line_width[0,k],ycenter[0,k],0],yindices) ,label="m1")
            # plt.plot(profile_model([A[1],line_width[1,k],ycenter[1,k],0],yindices),label="m2")
            # plt.plot(profile_model([A[2],line_width[2,k],ycenter[2,k],0],yindices),label="m3")
            # plt.legend()
            # plt.subplot(1,2,2)
            # plt.plot(rn+0*datacol,label="sig")
            # plt.plot(residuals[:,k]*rn,label="res")
            # plt.legend()
            # plt.show()
        else:
            residuals[:, k]=np.nan
            out[k,:]=np.nan

    return out,residuals

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    # x5, x4, x3, x2, x1, x0
    # -6.347647789728822e-23, -8.763050043241049e-18, -1.2137086057712525e-13, 9.833295632348853e-10, 2.205533800453195e-05, 2.438475434083729
    # -1.5156054451627633e-19, 8.603654109299864e-16, -1.80716247518748e-12, 2.238166690200087e-09, 2.1123747999061772e-05, 2.3625134057175177
    # 1.2043489185290055e-19, -6.07134563401576e-16, 1.017386463625886e-12, -9.26853752638751e-11, 2.1241653129345714e-05, 2.291160322496969
    # 3.968104754663601e-19, -1.3124938289815753e-15, 1.112571593986215e-12, 6.134905626295576e-10, 2.017316485869667e-05, 2.2242225934551327
    # 8.041271039507505e-20, -1.188553085654241e-15, 3.294776997678977e-12, -2.9632213749811383e-09, 2.1429663086447367e-05, 2.1608253921854454
    # 9.523371077287098e-20, 1.9255684438936675e-16, -1.7740532731738176e-12, 3.2497136280253542e-09, 1.785770822240155e-05, 2.10169669134709
    # -1.1921491225993055e-20, -2.4710800082806276e-17, -4.745847155094822e-13, 2.324820920356842e-09, 1.6978632482708987e-05, 2.0455151483483864
    # -1.0581425110234265e-20, 2.1172786131371726e-16, -7.916021064868458e-13, 1.4427033600147576e-09, 1.7857955925450578e-05, 1.9918890220516599
    # -7.67607806109257e-19, 3.893162857192925e-15, -6.697099419050439e-12, 5.105147905651209e-09, 1.6235110771773918e-05, 1.9414551546740444

    mykpicdir = "/scr3/jruffio/data/kpic/"
    # kap_And_dir = os.path.join(mykpicdir,"kap_And_20191107")
    # kap_And_dir = os.path.join(mykpicdir,"bet_Peg_20191108")
    # kap_And_dir = os.path.join(mykpicdir,"DH_Tau_20191108")
    kap_And_dir = os.path.join(mykpicdir,"kap_And_20191013")
    # kap_And_dir = os.path.join(mykpicdir,"2M0746A_20191012")
    # kap_And_dir = os.path.join(mykpicdir,"2M0746B_20191012")



    mode = "science"
    # mode = "star"
    # mode = "refstar"

    # background
    # try:
    if 1:
        background_med_filename = os.path.join(kap_And_dir,"calib","background_{0}_med.fits".format(mode.replace("refstar","star")))
        hdulist = pyfits.open(background_med_filename)
        background_med = hdulist[0].data
        background_badpixmap_filename = os.path.join(kap_And_dir, "calib", "background_{0}_badpixmap.fits".format(mode.replace("refstar","star")))
        hdulist = pyfits.open(background_badpixmap_filename)
        background_badpixmap = hdulist[0].data
    # except:
    #     background_med=np.zeros((2048,2048))
    #     background_badpixmap=np.ones((2048,2048))
    ny,nx = background_med.shape
    # plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.imshow(background60_med*background60_badpixmap,interpolation="nearest",origin="lower")
    # plt.subplot(1,2,2)
    # plt.imshow(background60_badpixmap,interpolation="nearest",origin="lower")
    # plt.show()

    star_filelist = glob(os.path.join(kap_And_dir, mode, "*.fits"))
    # print(star_filelist)
    # exit()
    #fibers: [1, 2, 3, 1, 2, 3, 2, 2, 2, 2, 2]
    # planet is fiber 2
    # star_filelist = ["/scr3/jruffio/data/kpic/kap_And_20191107/star/nspec191107_0041.fits",
    #                  "/scr3/jruffio/data/kpic/kap_And_20191107/star/nspec191107_0042.fits",
    #                  "/scr3/jruffio/data/kpic/kap_And_20191107/star/nspec191107_0043.fits"]

    fiberall = [[30,150],[220,330],[420,520],[640,720],[865,930],[1090,1170],[1320,1420],[1570,1690],[1840,1980]]

    fiber1 = [[70,150],[260,330],[460,520],[680,720],[900,930],[1120,1170],[1350,1420],[1600,1690],[1870,1980]]
    fiber2 = [[50,133],[240,320],[440,510],[650,710],[880,910],[1100,1150],[1330,1400],[1580,1670],[1850,1960]]
    fiber3 = [[30,120],[220,300],[420,490],[640,690],[865,890],[1090,1130],[1320,1380],[1570,1650],[1840,1940]]
    fibers = {1:fiber1,2:fiber2,3:fiber3}
    fiber1_template = np.zeros(2048)
    for x1,x2 in fiber1:
        fiber1_template[x1+10:x2-10] = 1
    fiber2_template = np.zeros(2048)
    for x1,x2 in fiber2:
        fiber2_template[x1+10:x2-10] = 1
    fiber3_template = np.zeros(2048)
    for x1,x2 in fiber3:
        fiber3_template[x1+10:x2-10] = 1
    fibers_template = {1: fiber1_template, 2: fiber2_template, 3: fiber3_template}
    

    fiber_list = []
    star_list = []
    for filename in star_filelist:
        hdulist = pyfits.open(filename)
        star_im = hdulist[0].data.T[:,::-1]
        print("MJD",hdulist[0].header["MJD"],hdulist[0].header["MJD"]+2400000.5)
        hdulist.close()

        star_im_skysub = star_im-background_med
        star_list.append(star_im_skysub)

        if mode == "science":
            fiber_list.append(2)
        elif "star" in mode:
            flattened = np.nanmean(star_im_skysub*background_badpixmap,axis=1)
            fiber_list.append(np.argmax([np.nansum(fiber1_template * flattened),
                               np.nansum(fiber2_template*flattened),
                               np.nansum(fiber3_template*flattened)])+1)

    star_cube = np.array(star_list)
    # exit()

    ##calculate traces, FWHM, stellar spec for each fibers
    # fiber,order,x,[y,yerr,FWHM,FHWMerr,flux,fluxerr],
    for star_im_id in range(star_cube.shape[0]):
        if 1:
            polyfit_trace_calib=np.zeros((3,9,2048,5))
            old_residuals=np.zeros((3,2048,2048))
            for fiber_num in np.arange(1, 4):
                out = os.path.join(kap_And_dir, "calib", "trace_calib_polyfit_fiber{0}.fits".format(fiber_num))
                if len(glob(out)) == 0:
                    continue
                hdulist = pyfits.open(out)
                polyfit_trace_calib[fiber_num-1,:,:,:] = hdulist[0].data
                out = os.path.join(kap_And_dir, "calib", "residuals_fiber{0}.fits".format(fiber_num))
                hdulist = pyfits.open(out)
                old_residuals[fiber_num-1,:,:] = hdulist[0].data
            print(polyfit_trace_calib.shape)
            print(old_residuals.shape)

            # background_badpixmap[np.where(np.abs(old_residuals)>3)]=np.nan

            star_im = star_cube[star_im_id]

            extract_paras = np.zeros((9,nx,6))
            residuals = np.zeros(star_im.shape)

            numthreads=30
            pool = mp.Pool(processes=numthreads)

            for order_id,(y1,y2) in enumerate(fiberall):
            # if 1:
            #     order_id, (y1, y2) = 8, fiberall[8]
                print(order_id,(y1,y2))
                yindices = np.arange(y1,y2)
                # star_stamp = star_im[x1:x2,:]
                # plt.imshow(star_stamp,origin="lower")
                # plt.show()
                # exit()

                if 0:
                    xindices = np.arange(700,2048)
                    out,residuals = _extract_flux((xindices,yindices,
                                                 star_im[y1:y2,xindices[0]:xindices[-1]+1],
                                                 background_badpixmap[y1:y2,xindices[0]:xindices[-1]+1],
                                                  polyfit_trace_calib[:,order_id,xindices[0]:xindices[-1]+1,1],
                                                  polyfit_trace_calib[:,order_id,xindices[0]:xindices[-1]+1,2]))
                    plt.subplot(1,2,1)
                    plt.imshow(star_im[y1:y2,xindices[0]:xindices[-1]+1])
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
                    w_chunks = []
                    y0_chunks = []
                    for k in range(N_chunks-1):
                        indices_chunks.append(np.arange(k*chunk_size,(k+1)*chunk_size))
                        data_chunks.append(star_im[y1:y2,k*chunk_size:(k+1)*chunk_size])
                        badpix_chunks.append(background_badpixmap[y1:y2,k*chunk_size:(k+1)*chunk_size])
                        w_chunks.append(polyfit_trace_calib[:,order_id,k*chunk_size:(k+1)*chunk_size,1])
                        y0_chunks.append(polyfit_trace_calib[:,order_id,k*chunk_size:(k+1)*chunk_size,2])
                    indices_chunks.append(np.arange((N_chunks-1)*chunk_size,nx))
                    data_chunks.append(star_im[y1:y2,(N_chunks-1)*chunk_size:nx])
                    badpix_chunks.append(background_badpixmap[y1:y2,(N_chunks-1)*chunk_size:nx])
                    w_chunks.append(polyfit_trace_calib[:,order_id,(N_chunks-1)*chunk_size:nx,1])
                    y0_chunks.append(polyfit_trace_calib[:,order_id,(N_chunks-1)*chunk_size:nx,2])
                    outputs_list = pool.map(_extract_flux, zip(indices_chunks,
                                                            itertools.repeat(yindices),
                                                            data_chunks,
                                                            badpix_chunks,
                                                            w_chunks,
                                                            y0_chunks))

                    normalized_psfs_func_list = []
                    chunks_ids = []
                    for xindices,out in zip(indices_chunks,outputs_list):
                        extract_paras[order_id,xindices[0]:xindices[-1]+1,:] = out[0]
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

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=extract_paras))
            out = os.path.join(kap_And_dir, "calib", os.path.basename(star_filelist[star_im_id]).replace(".fits","_"+mode+"flux_extract_f{0}.fits".format(fiber_list[star_im_id])))
            print(out)
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=residuals))
            out = os.path.join(kap_And_dir, "calib", os.path.basename(star_filelist[star_im_id]).replace(".fits","_"+mode+"flux_extract_res_f{0}.fits".format(fiber_list[star_im_id])))
            try:
                hdulist.writeto(out, overwrite=True)
            except TypeError:
                hdulist.writeto(out, clobber=True)
            hdulist.close()

    exit()