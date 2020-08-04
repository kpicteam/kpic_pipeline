from glob import glob
import os
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from utils.spectra import *

def get_err_from_posterior(x,posterior):
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior/np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    if len(x[0:np.min([argmax_post+1,len(x)])]) < 2:
        lx = x[0]
    else:
        tmp_cumpost = cum_posterior[0:np.min([argmax_post+1,len(x)])]
        tmp_x= x[0:np.min([argmax_post+1,len(x)])]
        deriv_tmp_cumpost = np.insert(tmp_cumpost[1::]-tmp_cumpost[0:np.size(tmp_cumpost)-1],np.size(tmp_cumpost)-1,0)
        try:
            whereinflection = np.where(deriv_tmp_cumpost<0)[0][0]
            where2keep = np.where((tmp_x<=tmp_x[whereinflection])+(tmp_cumpost>=tmp_cumpost[whereinflection]))
            tmp_cumpost = tmp_cumpost[where2keep]
            tmp_x = tmp_x[where2keep]
        except:
            pass
        lf = interp1d(tmp_cumpost,tmp_x,bounds_error=False,fill_value=x[0])
        lx = lf(1-0.6827)
    if len(x[argmax_post::]) < 2:
        rx=x[-1]
    else:
        tmp_cumpost = cum_posterior[argmax_post::]
        tmp_x= x[argmax_post::]
        deriv_tmp_cumpost = np.insert(tmp_cumpost[1::]-tmp_cumpost[0:np.size(tmp_cumpost)-1],np.size(tmp_cumpost)-1,0)
        try:
            whereinflection = np.where(deriv_tmp_cumpost>0)[0][0]
            where2keep = np.where((tmp_x>=tmp_x[whereinflection])+(tmp_cumpost>=tmp_cumpost[whereinflection]))
            tmp_cumpost = tmp_cumpost[where2keep]
            tmp_x = tmp_x[where2keep]
        except:
            pass
        rf = interp1d(tmp_cumpost,tmp_x,bounds_error=False,fill_value=x[-1])
        rx = rf(1-0.6827)
    return lx,x[argmax_post],rx,argmax_post


if __name__ == "__main__":
    try:
        import mkl

        mkl.set_num_threads(1)
    except:
        pass
    fontsize=12
    mykpicdir = "/scr3/jruffio/data/kpic/"
    out_pngs =  "/scr3/jruffio/data/kpic/figures/"
    if 0:
        target = "HIP_12787_A"
        rv_star,rverr_star  = 4.38,1.84
        vsini_star , vsinierr_star = np.nan,np.nan
        sciencedir = os.path.join(mykpicdir,"20191108_HIP_12787_A")
        filelist = glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        sciencedir = os.path.join(mykpicdir,"20191014_HIP_12787_A")
        filelist = filelist + glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        myfib = None
    if 0:
        target = "HIP_12787_B"
        rv_star,rverr_star  = 4.38,1.84
        vsini_star , vsinierr_star = np.nan,np.nan
        sciencedir = os.path.join(mykpicdir,"20191108_HIP_12787_B")
        filelist = glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        sciencedir = os.path.join(mykpicdir,"20191014_HIP_12787_B")
        filelist = filelist + glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        myfib = None
    if 0:
        target = "DH_Tau"
        rv_star,rverr_star  = 16.52,0.04
        vsini_star , vsinierr_star = 10.9,0.6
        sciencedir = os.path.join(mykpicdir,"20191108_DH_Tau")
        filelist = glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        sciencedir = os.path.join(mykpicdir,"20191215_DH_Tau")
        filelist = filelist + glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        myfib = None
    if 0:
        target = "DH_Tau_B"
        rv_star,rverr_star  = 16.52,0.04
        vsini_star , vsinierr_star = 10.9,0.6
        sciencedir = os.path.join(mykpicdir,"20191108_DH_Tau_B")
        filelist = glob(os.path.join(sciencedir, "out","flux_and_posterior.fits"))
        # sciencedir = os.path.join(mykpicdir,"20191215_DH_Tau_B")
        # filelist = filelist + glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        myfib = 2
    if 0:
        target = "2M0746A"
        rv_star,rverr_star  = 54.7,0.8
        vsini_star , vsinierr_star = 19,2
        sciencedir = os.path.join(mykpicdir,"20191012_2M0746A")
        filelist = glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        sciencedir = os.path.join(mykpicdir,"20191108_2M0746A")
        filelist = filelist + glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        myfib = 1
    if 1:
        target = "2M0746B"
        rv_star,rverr_star  = 54.7,0.8
        vsini_star , vsinierr_star = 19,2
        sciencedir = os.path.join(mykpicdir,"20191012_2M0746B")
        filelist = glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        sciencedir = os.path.join(mykpicdir,"20191108_2M0746B")
        filelist = filelist + glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        # print(filelist)
        # exit()
        myfib = 1
    if 0:
        target = "gg_Tau"
        rv_star,rverr_star  = 12.0,0.01
        vsini_star , vsinierr_star = np.nan,np.nan
        sciencedir = os.path.join(mykpicdir,"20191013B_gg_Tau")
        filelist = glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        # sciencedir = os.path.join(mykpicdir,"20191215_DH_Tau_B")
        # filelist = filelist + glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        myfib = 1
    if 0:
        target = "gg_Tau_B"
        rv_star,rverr_star  = 12.0,0.01
        vsini_star , vsinierr_star = np.nan,np.nan
        sciencedir = os.path.join(mykpicdir,"20191013B_gg_Tau_B")
        filelist = glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        # sciencedir = os.path.join(mykpicdir,"20191215_DH_Tau_B")
        # filelist = filelist + glob(os.path.join(sciencedir, "out","*_flux_and_posterior.fits"))
        myfib = 1
    filelist.sort()

    lb_rv_list = []
    rb_rv_list = []
    max_rv_list = []
    lb_vsini_list = []
    rb_vsini_list = []
    max_vsini_list = []

    date_list = []
    mjd_list = []
    fib_list = []
    framenum_list=[]

    linestyle_list = ["-","--",":"]
    color_list = ["#ff9900", "#0099cc", "#6600ff"]

    plt.figure(1,figsize=(9,4))
    print(filelist)
    # exit()
    for fileid,filename in enumerate(filelist):
        print(filename)
        hdulist = pyfits.open(filename)
        fluxout = hdulist[0].data
        print(fluxout.shape)
        logpostout = hdulist[1].data
        vsini_list = hdulist[2].data
        rv_list = hdulist[3].data
        post = np.exp(logpostout[0,:,:] - np.nanmax(logpostout[0,:,:]))
        try:
            combined_logpost = combined_logpost+logpostout
        except:
            combined_logpost = logpostout

        date = os.path.basename(filename).replace("nspec","").split("_")[0]
        date_list.append(date)
        framenum = os.path.basename(filename).replace("nspec","").split("_")[1]
        framenum_list.append(framenum)

        sciencefilename = os.path.join(os.path.dirname(filename), "..",
                                       os.path.basename(filename).replace("_flux_and_posterior.fits", ".fits"))
        hdulist = pyfits.open(sciencefilename)
        mymjd = float(hdulist[0].header["MJD"])
        mjd_list.append(mymjd)
        if myfib is None:
            science_spec,science_err,slit_spec,dark_spec,science_baryrv = combine_spectra_from_folder([sciencefilename],"science")
            fib = np.nanargmax(np.nanmean(science_spec, axis=(1, 2)))
            fib_list.append(fib)
        else:
            fib_list.append(myfib)

        plt.subplot(1, 2, 1)
        rvpost = np.nansum(post, axis=0)
        plt.plot(rv_list, rvpost / np.nanmax(rvpost),label=os.path.basename(filename).replace("_fluxes_flux_and_posterior.fits",""),linestyle=linestyle_list[myfib],color=color_list[myfib])
        plt.xlabel("RV")
        lb_rv,max_rv,rb_rv,_ = get_err_from_posterior(rv_list,rvpost/ np.nanmax(rvpost))
        lb_rv_list.append(lb_rv)
        rb_rv_list.append(rb_rv)
        max_rv_list.append(max_rv)
        plt.subplot(1, 2, 2)
        vsinipost = np.nansum(post, axis=1)
        plt.plot(vsini_list, vsinipost / np.nanmax(vsinipost),label=os.path.basename(filename).replace("_fluxes_flux_and_posterior.fits",""),linestyle=linestyle_list[myfib],color=color_list[myfib])
        plt.xlabel("vsin(i)")
        lb_vsini,max_vsini,rb_vsini,_ = get_err_from_posterior(vsini_list,vsinipost)
        lb_vsini_list.append(lb_vsini)
        rb_vsini_list.append(rb_vsini)
        max_vsini_list.append(max_vsini)
        # plt.show()
    date_list = np.array(date_list)
    fib_list = np.array(fib_list)
    framenum_list = np.array(framenum_list)
    mjd_list = np.array(mjd_list)

    max_rv_list = np.array(max_rv_list)
    lerr_rv_list = np.array(max_rv_list) - np.array(lb_rv_list)
    rerr_rv_list = np.array(rb_rv_list) - np.array(max_rv_list)
    max_vsini_list = np.array(max_vsini_list)
    lerr_vsini_list = np.array(max_vsini_list) - np.array(lb_vsini_list)
    rerr_vsini_list = np.array(rb_vsini_list) - np.array(max_vsini_list)

    combined_post = np.exp(combined_logpost[0, :, :] - np.nanmax(combined_logpost[0, :, :]))
    combined_rvpost = np.nansum(combined_post, axis=0)
    lb_rv,max_rv,rb_rv,_ = get_err_from_posterior(rv_list,combined_rvpost/ np.nanmax(combined_rvpost))
    max_rv = np.array(max_rv)
    lerr_rv = np.array(max_rv) - np.array(lb_rv)
    rerr_rv = np.array(rb_rv) - np.array(max_rv)
    print(lerr_rv_list)
    print(rerr_rv_list)
    print(max_rv,lerr_rv,rerr_rv)

    plt.subplot(1, 2, 1)
    plt.xlabel("RV (km/s)",fontsize=fontsize)
    plt.xlim([40,60])
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="center left",fontsize=fontsize)
    # plt.legend()
    plt.subplot(1, 2, 2)
    plt.xlabel("vsin(i) (km/s)",fontsize=fontsize)
    plt.xlim([0,40])
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)

    if 1:
        print("Saving "+os.path.join(out_pngs,target+"_rv_vsini_post.pdf"))
        plt.savefig(os.path.join(out_pngs,target+"_rv_vsini_post.pdf"))
        plt.savefig(os.path.join(out_pngs,target+"_rv_vsini_post.png"))

    plt.figure(2,figsize=(12,4))
    for linestyle,color,fib in zip(linestyle_list,color_list,np.arange(3)):
        where_fib = np.where(fib==fib_list)
        # plt.errorbar(np.arange(len(max_rv_list)),max_rv_list,yerr=[lerr_rv_list,rerr_rv_list ],fmt="none",color="#ff9900")
        eb1 = plt.errorbar(np.arange(len(max_rv_list))[where_fib],max_rv_list[where_fib],
                     yerr=[lerr_rv_list[where_fib], rerr_rv_list[where_fib]],fmt="none",color=color,label="Fiber {0}".format(fib))
        eb1[-1][0].set_linestyle(linestyle)
        plt.plot(np.arange(len(max_rv_list))[where_fib],max_rv_list[where_fib],"x",color=color) # #0099cc  #ff9900

    plt.fill_between([0,len(max_rv_list)],rv_star-rverr_star,rv_star+rverr_star,alpha=0.3,color="grey",label="Litterature")

    plt.ylim([rv_list[0],rv_list[-1]])
    plt.xlim([0,len(max_rv_list)+1])
    unique_dates = np.unique(date_list)
    for date in unique_dates:
        where_date = np.where(date_list==date)
        first = where_date[0][0]
        last = where_date[0][-1]
        plt.annotate("",xy=(first,np.nanmin(max_rv_list-lerr_rv_list)),xytext=(last+0.001,np.nanmin(max_rv_list-lerr_rv_list)),xycoords="data",arrowprops={'arrowstyle':"|-|",'shrinkA':0.1,'shrinkB':0.1})
        plt.gca().text((first+last)/2.,np.nanmin(max_rv_list-lerr_rv_list),date[2:4]+"-"+date[4:6]+"-20"+date[0:2],ha="left",va="top",rotation=-60,size=fontsize)
        plt.gca().text(first,max_rv_list[first]+rerr_rv_list[first],"$\Delta$t=0 min",ha="center",va="bottom",rotation=0,size=fontsize)
        for k,framenum in zip(np.arange(len(framenum_list))[where_date[0][1::]],framenum_list[where_date[0][1::]]):
            delta_time = (mjd_list[k]- mjd_list[first])*(24.*60)
            plt.gca().text(k,max_rv_list[k]+rerr_rv_list[k],"{0:.0f}".format(delta_time),ha="center",va="bottom",rotation=0,size=fontsize)
            # plt.gca().text(k,max_rv_list[k]+rerr_rv_list[k],"{0}".format(framenum),ha="center",va="bottom",rotation=0,size=fontsize)

    plt.gca().text(len(max_rv_list),np.nanmax(max_rv_list+rerr_rv_list),target,ha="right",va="top",rotation=0,size=fontsize*1.5)
    plt.xlim([-1,len(max_rv_list)])
    _min,_max = np.nanmin(max_rv_list-lerr_rv_list),np.nanmax(max_rv_list+rerr_rv_list)
    plt.ylim([_min-(_max-_min)*(0.25),_max])
    plt.ylabel("RV (km/s)",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)#set_position(("data",0))
    plt.tick_params(axis="x",which="both",bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="upper left",bbox_to_anchor=[1.0,1.0],frameon=True,fontsize=fontsize)
    # plt.show()
    plt.tight_layout()

    if 1:
        print("Saving "+os.path.join(out_pngs,target+"_rv.pdf"))
        plt.savefig(os.path.join(out_pngs,target+"_rv.pdf"))
        plt.savefig(os.path.join(out_pngs,target+"_rv.png"))

    plt.figure(3,figsize=(12,4))
    for linestyle,color,fib in zip(linestyle_list,color_list,np.arange(3)):
        where_fib = np.where(fib==fib_list)
        # plt.errorbar(np.arange(len(max_rv_list)),max_rv_list,yerr=[lerr_rv_list,rerr_rv_list ],fmt="none",color="#ff9900")
        eb1 = plt.errorbar(np.arange(len(max_vsini_list))[where_fib],max_vsini_list[where_fib],
                     yerr=[lerr_vsini_list[where_fib], rerr_vsini_list[where_fib]],fmt="none",color=color,label="Fiber {0}".format(fib))
        eb1[-1][0].set_linestyle(linestyle)
        plt.plot(np.arange(len(max_vsini_list))[where_fib],max_vsini_list[where_fib],"x",color=color) # #0099cc  #ff9900

    plt.fill_between([0,len(max_vsini_list)],vsini_star-vsinierr_star,vsini_star+vsinierr_star,alpha=0.3,color="grey",label="Litterature")

    plt.ylim([rv_list[0],rv_list[-1]])
    plt.xlim([0,len(max_vsini_list)+1])
    unique_dates = np.unique(date_list)
    for date in unique_dates:
        where_date = np.where(date_list==date)
        first = where_date[0][0]
        last = where_date[0][-1]
        plt.annotate("",xy=(first,np.nanmin(max_vsini_list-lerr_vsini_list)),xytext=(last+0.001,np.nanmin(max_vsini_list-lerr_vsini_list)),xycoords="data",arrowprops={'arrowstyle':"|-|",'shrinkA':0.1,'shrinkB':0.1})
        plt.gca().text((first+last)/2.,np.nanmin(max_vsini_list-lerr_vsini_list),date[2:4]+"-"+date[4:6]+"-20"+date[0:2],ha="left",va="top",rotation=-60,size=fontsize)
        plt.gca().text(first,max_vsini_list[first]+rerr_vsini_list[first],"$\Delta$t=0 min",ha="center",va="bottom",rotation=0,size=fontsize)
        for k,framenum in zip(np.arange(len(framenum_list))[where_date[0][1::]],framenum_list[where_date[0][1::]]):
            delta_time = (mjd_list[k]- mjd_list[first])*(24.*60)
            plt.gca().text(k,max_vsini_list[k]+rerr_vsini_list[k],"{0:.0f}".format(delta_time),ha="center",va="bottom",rotation=0,size=fontsize)
            # plt.gca().text(k,max_vsini_list[k]+rerr_vsini_list[k],"{0}".format(framenum),ha="center",va="bottom",rotation=0,size=fontsize)


    plt.gca().text(len(max_vsini_list),np.nanmax(max_vsini_list+rerr_vsini_list),target,ha="right",va="top",rotation=0,size=fontsize*1.5)
    plt.xlim([-1,len(max_vsini_list)])
    _min,_max = np.nanmin(max_vsini_list-lerr_vsini_list),np.nanmax(max_vsini_list+rerr_vsini_list)
    plt.ylim([_min-(_max-_min)*(0.25),_max])
    plt.ylabel("v sin(i) (km/s)",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)#set_position(("data",0))
    plt.tick_params(axis="x",which="both",bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="upper left",bbox_to_anchor=[1.0,1.0],frameon=True,fontsize=fontsize)
    plt.tight_layout()

    if 1:
        print("Saving "+os.path.join(out_pngs,target+"_vsini.pdf"))
        plt.savefig(os.path.join(out_pngs,target+"_vsini.pdf"))
        plt.savefig(os.path.join(out_pngs,target+"_vsini.png"))
    plt.show()