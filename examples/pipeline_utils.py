from glob import glob
import os
import numpy as np
import astropy.io.fits as fits
import pandas as pd 
from kpicdrp.caldb import det_caldb, trace_caldb
import kpicdrp.data as data 
import kpicdrp.trace as trace
import kpicdrp.extraction as extraction
from kpicdrp import throughput 
import matplotlib.pyplot as plt

def get_filenums(range_info):
    filenums = np.array([], dtype=int)
    for seq in range_info:
        this_start = seq[0]
        this_stop = seq[1]
        this_files = np.arange(this_start, this_stop+1)
        filenums = np.append(filenums, this_files)

    print(str(len(filenums)) + ' of files identified:')
    print(filenums)

    return filenums

def get_filelist(raw_dir, filenums, filestr, raw_datadir):
    filelist = glob(os.path.join(raw_dir, "*.fits"))

    print(len(filelist), len(filenums))

    # just return the files if nodsub
    if 'raw_pairsub' in raw_dir:
        return filelist

    elif len(filelist) != len(filenums):
        for fnum in filenums:
            fname = os.path.join(raw_datadir, filestr.format(fnum))
            print('Moving '  + fname)
            command = "rsync -a -u --ignore-existing " + fname + " " + raw_dir
            os.system(command)
        
        # remake filelist if this is the case
        #filelist = glob(os.path.join(raw_dir, "*.fits"))

        # only get files corresponding to input filenums
        filelist = [ os.path.join(raw_dir, filestr.format(i)) for i in filenums]

    print(len(filelist))

    return filelist

def parse_header_night(raw_dir):
    """
    read headers for all files in a given directory
    Args:
        raw_dir: directory of NIRSPEC files to analyze
    Returns:
        unique_targets: unique targets
        info_dict: dictionary of useful parameters for each file. Includes: target name, elevation, exptime, offset, SF# etc.
    """

    all_target, all_targetold, all_utdate, all_uttime, all_exptime, all_filename = [], [], [], [], [], []
    all_elev, all_airmass = [], []
    all_sf, all_mask = [], []
    all_dar, all_offset, all_fnum, all_echlpos, all_disppos, all_filt = [], [], [], [], [], []

    for file in glob(raw_dir + '/*.fits'):
        this_hdr = fits.open(file)[0].header

        # all kinds of names non-science frames might have
        if this_hdr['TARGNAME'] != 'HORIZON STOW' and this_hdr['TARGNAME'] != 'unknown' and this_hdr['TARGNAME'] != 'FOUL WEATHER' and this_hdr['TARGNAME'] != '':
            
            # only K band for now
            # if this_hdr['FILTER'] == 'Kband-new':
            # clean up names to conform to hcig1 convention
            targname_orig = this_hdr['TARGNAME']
            targname_new = targname_orig.replace('-', '')
            if 'Her' in targname_orig:  # for ups_Her
                targname_new = targname_new.replace(' ', '_')
            else:
                targname_new = targname_new.replace(' ', '')
            targname_new = targname_new.replace('HIP0', 'HIP')
            
            all_target.append(targname_new)
            all_targetold.append(targname_orig)

            all_utdate.append(this_hdr['DATE-OBS'])
            all_uttime.append(this_hdr['UT'])

            all_filename.append(file)
            all_fnum.append(this_hdr['FRAMENUM'])
            all_exptime.append(this_hdr['TRUITIME'])
            all_elev.append(this_hdr['EL'])
            all_airmass.append(this_hdr['AIRMASS'])
            all_sf.append(this_hdr['FIUGNM'])
            all_offset.append(this_hdr['FIUDSEP'])
            all_dar.append(this_hdr['FIUDAR'])
            all_mask.append(this_hdr['FIUCGNAM'])
            all_filt.append(this_hdr['FILTER'])
            all_echlpos.append(this_hdr['ECHLPOS'])
            all_disppos.append(this_hdr['DISPPOS'])

    dict_keys = ['FILENUM', 'TARGNAME', 'UTDATE', 'UTTIME', 'TRUITIME', 'EL', 'SFNUM', 'FIUDSEP', 'DAR', 'AIRMASS', 
    'FILEPATH', 'FILTER', 'CORONAGRAPH', 'ECHLPOS', 'DISPPOS', 'TARGNAME_ORIG']
    info_df = pd.DataFrame(list(zip(all_fnum, all_target, all_utdate, all_uttime, all_exptime, all_elev, all_sf,
                         all_offset, all_dar, all_airmass, all_filename, all_filt, all_mask, all_echlpos, all_disppos, all_targetold)),
                        columns=dict_keys)

    unique_targets = np.unique(all_target)

    return unique_targets, info_df

def do_extract_1d(filelist, out_flux_dir, out_filenames, use_nod_sub=True, mypool=None, bad_pixel_fraction=0.01):
    
    raw_sci_dataset = data.Dataset(filelist=filelist, dtype=data.DetectorFrame)

    # no need to load these for nod sub
    if not use_nod_sub:
        bkgd = det_caldb.get_calib(raw_sci_dataset[0], type="Background")
        badpixmap = det_caldb.get_calib(raw_sci_dataset[0], type="BadPixelMap")

    trace_dat = trace_caldb.get_calib(raw_sci_dataset[0])

    # get background traces if they aren't there already
    if 'b1' not in trace_dat.labels:
        trace_dat = trace.get_background_traces(trace_dat)

    # if using nod subtracted raw 2D frame, just load them
    if use_nod_sub:
        # this needs to be the nodsub.fits files
        sci_dataset = data.Dataset(filelist=filelist, dtype=data.DetectorFrame)
        print('Loading nodsub 2D frames directly.')
        fit_background = False  # do not fit background for nod sequences
    else:
        sci_dataset = extraction.process_sci_raw2d(raw_sci_dataset, bkgd, badpixmap, detect_cosmics=True, add_baryrv=True)
        print('Extracting 2D frame.')
        fit_background = True

    # extract the 1D fluxes
    spectral_dataset = extraction.extract_flux(sci_dataset, trace_dat, fit_background=fit_background,
     bad_pixel_fraction=bad_pixel_fraction, pool=mypool)

    spectral_dataset.save(filedir=out_flux_dir, filenames=out_filenames)

    return spectral_dataset, trace_dat

def do_nod_subtract(filelist, sub_mode, fiber_goals, out_nod_dir):

    raw_sci_dataset = data.Dataset(filelist=filelist, dtype=data.DetectorFrame)

    # nod mode needs only bad pixel map
    badpixmap = det_caldb.get_calib(raw_sci_dataset[0], type="BadPixelMap")

    # two choices for subtraction_mode
    # 'nod' is better if you the background is unchanging. 'pair' is better if one pair has better backgroud similarity
    print('subtraction mode is ' + sub_mode)
    sci_dataset = extraction.process_sci_raw2d(raw_sci_dataset, None, badpixmap, detect_cosmics=True, add_baryrv=True, nod_subtraction=sub_mode, fiber_goals=fiber_goals)

    sci_dataset.save(filedir=out_nod_dir)

    nodsub_frames = glob(os.path.join(out_nod_dir, "*.fits"))
    return nodsub_frames

def plot_spec(trace_dat, out_filenames, out_flux_dir, filenums, target_date_dir, show_plot=False):
    sf_indices = trace_dat.get_sci_indices()
    # print(filenums)
    for filename, filenum in zip(out_filenames, filenums):
        this_outfile = os.path.join(out_flux_dir, filename)
        # print(this_outfile)
        # load the 1D fluxes we just saved
        this_spectrum = data.Spectrum(filepath=this_outfile)
        spec = this_spectrum.fluxes
        # err = this_spectrum.errs
        for fib_id in sf_indices:
            fig = plt.figure(fib_id + 1, figsize=(12, 12))
            plt.title('SF ' + str(fib_id+1))
            for order_id in range(spec.shape[1]):
                plt.subplot(spec.shape[1], 1, order_id + 1)
                xs = np.arange(spec[fib_id, order_id, :].shape[0])
                plt.plot(xs, spec[fib_id, order_id, :], label=str(filenum) )

            plt.legend()
            plt.savefig(target_date_dir + '/SF' + str(fib_id+1) + '_extracted_spec.png', dpi=50)

    if show_plot:
        plt.show()
        plt.close('all')
    else:
        plt.close('all')

def run_nod(target_files, target_date_dir, inds, sfnum, out_flux_dir, existing_frames, mypool=None, show_plot=False):

    these_frames = target_files[inds]
    
    nodsub_dir = os.path.join(target_date_dir, 'raw_pairsub/')
    if not os.path.exists(nodsub_dir):
        os.makedirs(nodsub_dir)

    # first do nod subtraction. Somewhat arbitrary. MAKE IT SMARTER??
    if len(inds) < 12:
        sub_mode = 'nod'  # good default, pair requires e.g. 23232323 sequences
    else:
        sub_mode = 'pair'

    if 'HD984B' in target_date_dir:
        sub_mode == 'pair'

    # do nod sub on all available frames
    fiber_goals = [ int(s[-1]) for s in sfnum[inds] ]  # pull out sf numbers
    print(fiber_goals, these_frames)
    print('Running nod subtraction on all frames')
    nodsub_frames = do_nod_subtract(these_frames, sub_mode, fiber_goals, nodsub_dir)

    all_frames = [ os.path.join(out_flux_dir, f.split('raw_pairsub/')[1].replace('_nodsub.fits', '_nodsub_spectra.fits') ) for f in nodsub_frames]
    
    # extract new frames only
    new_inds = np.where(np.isin(all_frames, existing_frames) == False)[0]  # pick out indices of new frames
    new_frames = np.asarray(nodsub_frames)[new_inds]
    out_filenames = [f.split('raw_pairsub/')[1].replace('_nodsub.fits', '_nodsub_spectra.fits') for f in new_frames]
    filenums = [f[-9:].split('.fits')[0] for f in these_frames[new_inds]]

    # print(new_inds)
    # print(len(new_frames), len(out_filenames), len(filenums))
    # print(new_frames, out_filenames, filenums)

    # extract!
    spectral_dataset, trace_dat = do_extract_1d(new_frames, out_flux_dir, out_filenames, use_nod_sub=True, mypool=mypool)
    
    plot_spec(trace_dat, out_filenames, out_flux_dir, filenums, target_date_dir, show_plot=show_plot)

    return out_filenames

def run_bkgd(these_frames, target_date_dir, out_flux_dir, existing_frames, mypool=None, show_plot=False):
    out_filenames = [f.split('spec/')[1].replace('.fits', '_bkgdsub_spectra.fits') for f in these_frames]

    # extract new frames only
    all_frames = [os.path.join(out_flux_dir, f) for f in out_filenames]
    new_inds = np.where(np.isin(all_frames, existing_frames) == False)[0]  # pick out indices of new frames
    new_frames = these_frames[new_inds]
    print(new_frames)

    filenums = [f[-9:].split('.fits')[0] for f in new_frames]
    new_out_filenames = np.asarray(out_filenames)[new_inds]

    spectral_dataset, trace_dat = do_extract_1d(new_frames, out_flux_dir, new_out_filenames, use_nod_sub=False, mypool=mypool)
    plot_spec(trace_dat, new_out_filenames, out_flux_dir, filenums, target_date_dir, show_plot=show_plot)

    return out_filenames

def make_dirs_target(kpicdir, target_name, obsdate):
    target_main_dir = os.path.join(kpicdir, "science", target_name)
    if not os.path.exists(target_main_dir):
        os.makedirs(target_main_dir)
    target_date_dir = os.path.join(kpicdir, "science", target_name, obsdate)
    if not os.path.exists(target_date_dir):
        os.makedirs(target_date_dir)

     # Testing now, so folder name is different
    out_flux_dir = os.path.join(target_date_dir, "jxuan_fluxes")
    if not os.path.exists(os.path.join(out_flux_dir)):
        os.makedirs(os.path.join(out_flux_dir))

    target_raw = os.path.join(target_date_dir, "raw")
    if not os.path.exists(os.path.join(target_raw)):
        os.makedirs(os.path.join(target_raw))

    return target_date_dir, out_flux_dir, target_raw

def rsync_files(these_frames, target_raw):
    for fname in these_frames:
        # print('Moving '  + fname)
        command = "rsync -a -u --ignore-existing " + fname + " " + target_raw
        os.system(command)

# this is still manual right now, reading the logs
def save_bad_frames(obsdate, df, df_path):
    bad_frames = []
    if obsdate == '20220723':
        bad = [197,177,178,179,180,181]
        saturated = np.linspace(153, 164, 12, dtype=int)
        l_band = np.linspace(441, 468, 28, dtype=int)
        bad_frames = np.append(np.asarray(bad), saturated)
        bad_frames = np.append(bad_frames, l_band)

    elif obsdate == '20220722':
        bad_frames = [197,198,199]

    elif obsdate == '20220718':
        bad = np.linspace(87, 146, 60, dtype=int)
        bad_frames = np.append(bad, np.array([177,]))

    ## Add if there are bad frames for a date

    # initialize all to 0 (good)
    df.insert(10, "BADFRAME", '')
    df['BADFRAME'] = 0
    
    # bad frames have value 1
    for b in bad_frames:
        df.loc[df['FILENUM'] == b, 'BADFRAME'] = 1

    # just overwrite the previous file
    df.to_csv(df_path, index=False)

    return df

def get_throughput(path, wv_soln, k_mag, bb_temp=7000, show_plot=False):

    spec = data.Spectrum(filepath=path)
    spec.calibrate_wvs(wv_soln) 

    all_wvs = spec._wvs

    peak_thru, thru_all_orders = throughput.calculate_peak_throughput(spec, k_mag, bb_temp=bb_temp, fib=None, plot=show_plot)

    return peak_thru, thru_all_orders, all_wvs

def add_spec_column(df, out_filenames, these_frames, out_flux_dir):

    # initialize the columns
    if not 'SPECFILE' in df.columns:
        df.insert(14, "SPECFILE", '')
        df.insert(13, "SPECDIR", '')

    # add output spectra name
    for out_f, raw_f in zip(out_filenames, these_frames):
        df.loc[df['FILEPATH'] == raw_f, 'SPECFILE'] = out_f
        df.loc[df['FILEPATH'] == raw_f, 'SPECDIR'] = out_flux_dir

    return df

def add_thru_column(df, these_frames, all_thru95):

    # initialize the columns
    if not 'THRU95' in df.columns:
        df.insert(16, "THRU95", '')
        # df.insert(17, "THRUARRAY", '')  
        # df.insert(18, "WVS", '')  # we only have 1 WVS per night

    # add output spectra name
    for thru95, raw_f in zip(all_thru95, these_frames):
        print(thru95)
        df.loc[df['SPECFILE'] == raw_f, 'THRU95'] = thru95

    return df

def read_user_options(opts):
    
    if len(opts) == 0:
        return False, False, False

    #print(opts)
    nod_only_boo, new_files_boo, show_plot = False,False,False

    try:
        nod_only = opts[0][1]
    except:
        nod_only = 'n'
    
    if nod_only == 'y':
        nod_only_boo = True
        print('Nod subtraction only (requires more than 1 SF with data, and they have the same tint).')
    
    try:
        new_files = opts[1][1]
    except:
        new_files = 'n'
    if new_files == 'y':
        new_files_boo = True
        print('Rescanning directory to include new files.')

    try:
        plot = opts[2][1]
    except:
        plot = 'n'
    if plot == 'y':
        show_plot = True
        print('Plotting spectra.')

    # print(nod_only, new_files, plot)
    return nod_only_boo, new_files_boo, show_plot

# Add when we discover more planets!
# Can we be smarter about this? mostly complicated by systems with more than 1 planet
def make_comp_dir(target_name, kpicdir, obsdate):
    if target_name == 'HR8799' or target_name == 'HD206893' or target_name == '51Eri' or target_name == 'PDS70':
        which_planet = input('Enter planet name for '+target_name+' (b (or B), c, d, e?) >>> ')
        which_planet = which_planet.strip()
        comp_name = target_name + which_planet
    else:
        comp_name = target_name + 'B'
    comp_date_dir, out_flux_dir, target_raw = make_dirs_target(kpicdir, comp_name, obsdate)

    return comp_date_dir, out_flux_dir, target_raw, comp_name