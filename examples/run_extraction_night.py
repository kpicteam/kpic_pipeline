import numpy as np
import os
import multiprocessing as mp
from glob import glob
import pandas as pd
import getopt
import sys
from pipeline_utils import parse_header_night, do_extract_1d, run_nod, plot_spec, make_dirs_target, rsync_files, save_bad_frames, add_spec_column, read_user_options
import warnings
warnings.filterwarnings("ignore")

# option to only do nod subtraction. Requires targets have data on more than 1 SF. 
# Parse user command line input
try:
    (opts, args) = getopt.getopt(sys.argv[1:], "hi", ["help",
                        "nod_only=", "new_files=", "plot="])
except getopt.GetoptError as err:
    print(str(err)) # will print something like "option -a not recognized"
    sys.exit(2)

# show_plot of extracted spectra? if True, extraction will pause after each target
nod_only, new_files, show_plot = read_user_options(opts)

# UT Date of observation
obsdate = input("Enter UT Date (e.g.20220723) >>> ")
obsdate = obsdate.strip()

# Things to change
#----------------------------------------------------------------
# main data dir
kpicdir = "/scr3/kpic/KPIC_Campaign/"
# where summary tables are stored
df_dir = os.path.join(kpicdir, 'nightly_tables')
# where raw data is stored. This should be a copy of the spec/ folder from Keck
# So you should probably only change "/scr3/kpic/Data/""
raw_datadir = os.path.join("/scr3/kpic/Data/", obsdate[2:], "spec")
# Adjust to your machine
mypool = mp.Pool(16)
#----------------------------------------------------------------

# targets_to_run = ['HIP95771', 'HD984', 'HIP115119']

# nspec file convention
filestr = "nspec"+obsdate[2:]+"_0{0:03d}.fits"

df_path = os.path.join(df_dir, obsdate+'.csv')
if new_files:
    unique_targets, orig_night_df = parse_header_night(raw_datadir)
    orig_night_df.to_csv(df_path, index=False)
else:
    # Try to load the night_df, if exists
    if os.path.exists(df_path):
        orig_night_df = pd.read_csv(df_path)
        print('Loaded existing nightly df.')
    else:
        # generate a big df for all files from this night.
        unique_targets, orig_night_df = parse_header_night(raw_datadir)
        orig_night_df.to_csv(df_path, index=False)

if not 'BADFRAME' in orig_night_df.columns:
    orig_night_df = save_bad_frames(obsdate, orig_night_df, df_path)

# return night_df without badframes. For indexing.
night_df = orig_night_df[orig_night_df['BADFRAME'] == 0]
print(str(night_df.shape[0]) + ' number of files to extract for this night.')

# remake unique targets. Sometimes all frames for a target are bad...
print('Targets observed (with good frames) were: ')
unique_targets =np.unique(night_df['TARGNAME'].values)
print(unique_targets)
# print(night_df.head())

# iterate over target
for target_name in unique_targets:
    print('Onto ' + target_name)
    # make folders for the target
    target_date_dir, out_flux_dir, target_raw = make_dirs_target(kpicdir, target_name, obsdate)

    # which files belong to this target
    target_files = night_df.loc[night_df['TARGNAME'] == target_name, 'FILEPATH'].values

    # other useful columns
    dist_sep = night_df.loc[night_df['TARGNAME'] == target_name, 'FIUDSEP'].values
    exptime = night_df.loc[night_df['TARGNAME'] == target_name, 'TRUITIME'].values
    sfnum = night_df.loc[night_df['TARGNAME'] == target_name, 'SFNUM'].values
    cgname = night_df.loc[night_df['TARGNAME'] == target_name, 'CORONAGRAPH'].values

    # companion or on-axis
    # using 35 mas, to handle cases where we intentionally offset RV star by 30 mas (prevent saturation...)
    # has to be pupil_mask. And condition
    on_axis_ind = np.where( (dist_sep < 35) & (cgname == 'pupil_mask'))[0]
    # sometimes, we use custom for dichroic out 
    if len(on_axis_ind) == 0:
        on_axis_ind = np.where( (dist_sep < 35) & (cgname == 'Custom'))[0]
    # or if still 0, could be MDA = apodizer
    if len(on_axis_ind) == 0:
        on_axis_ind = np.where( (dist_sep < 35) & (cgname == 'apodizer'))[0]

    # VFN mode + off-axis, for a companion. Or condition
    off_axis_ind = np.where( (dist_sep >= 35) | (cgname == 'vortex'))[0]

    # first check if we used more than 1 SF. If not, must do background subtraction
    print(np.unique(sfnum))
    if len(np.unique(sfnum)) == 1:
        bkgd_ind = np.where(exptime > -1)[0]  # everything
        nod_ind = np.array([])
    else:
        if nod_only:
            nod_ind = np.where(exptime > -1)[0]  # everything
            bkgd_ind = np.array([])  # empty for bkgd ind
        else:
            bkgd_ind = np.where(exptime < 15)[0]
            nod_ind = np.where(exptime >= 15)[0]  # To do: this breaks if there is only 1 SF

    # return indices of intersections between on/off axis and exptime
    # Case 1: bkgd and on-axis
    bkgd_onaxis = np.intersect1d(bkgd_ind, on_axis_ind)
    # Case 2: nod and on-axis
    nod_onaxis = np.intersect1d(nod_ind, on_axis_ind)
    # Case 3: nod and off-axis
    nod_offaxis = np.intersect1d(nod_ind, off_axis_ind)
    # Case 4: bkgd and off-axis - this is rare after 2021
    # also previous data won't have headers with all the info needed to run this pipeline
    
    print(nod_offaxis, nod_onaxis, bkgd_onaxis)

    if len(bkgd_onaxis) > 0:
        these_frames = target_files[bkgd_onaxis]
        # move the frames to raw dir
        rsync_files(these_frames, target_raw)
        out_filenames = [f.split('spec/')[1].replace('.fits', '_bkgdsub_spectra.fits') for f in these_frames]
        existing_frames = glob(os.path.join(out_flux_dir, "*.fits"))

        # print(len(existing_frames), len(these_frames))
        # if len(existing_frames) == 0:
        if len(existing_frames) < len(these_frames):
            print('Extracting on-axis flux for ' + target_name + ' from ' + obsdate + '; bkgd sub')
            # extract new frames only
            all_frames = [os.path.join(out_flux_dir, f) for f in out_filenames]
            new_inds = np.where(np.isin(all_frames, existing_frames) == False)[0]  # pick out indices of new frames
            new_frames = these_frames[new_inds]
            print(new_frames)

            filenums = [f[-9:].split('.fits')[0] for f in new_frames]
            new_out_filenames = np.asarray(out_filenames)[new_inds]

            spectral_dataset, trace_dat = do_extract_1d(new_frames, out_flux_dir, new_out_filenames, use_nod_sub=False, mypool=mypool)
            plot_spec(trace_dat, new_out_filenames, out_flux_dir, filenums, target_date_dir, show_plot=show_plot)

        else:
            print(target_name + ' has already been extracted.')
        orig_night_df = add_spec_column(orig_night_df, out_filenames, these_frames, out_flux_dir)

    if len(nod_onaxis) > 0:
        these_frames = target_files[nod_onaxis]
        rsync_files(these_frames, target_raw)

        existing_frames = glob(os.path.join(out_flux_dir, "*.fits"))
        # print(len(existing_frames), len(these_frames))
        if len(existing_frames) < len(these_frames):
            print('Extracting on-axis flux for ' + target_name + ' from ' + obsdate + '; nodding')
            out_filenames = run_nod(target_files, target_date_dir, nod_onaxis, sfnum, out_flux_dir, existing_frames, mypool=mypool, show_plot=show_plot)
        
        else:
            print(target_name + ' has already been extracted.')
            _out_filenames = glob(os.path.join(out_flux_dir, "*.fits"))
            out_filenames = [p.replace(out_flux_dir+'/', '') for p in _out_filenames]

        orig_night_df = add_spec_column(orig_night_df, out_filenames, these_frames, out_flux_dir)

    # for companion, append a letter to directory names.
    if len(nod_offaxis) > 0:
        # Smarter about this?
        if target_name == 'HR8799' or target_name == 'HD206893' or target_name == '51Eri' or target_name == 'PDS70':
            which_planet = input('Enter planet name for '+target_name+' (b (or B), c, d, e?) >>> ')
            which_planet = which_planet.strip()
            comp_name = target_name + which_planet
        else:
            comp_name = target_name + 'B'
        comp_date_dir, out_flux_dir, target_raw = make_dirs_target(kpicdir, comp_name, obsdate)

        these_frames = target_files[nod_offaxis]
        rsync_files(these_frames, target_raw)

        existing_frames = glob(os.path.join(out_flux_dir, "*.fits"))
        # print(len(existing_frames), len(these_frames))
        if len(existing_frames) < len(these_frames):
            print('Extracting off-axis flux for ' + comp_name + ' from ' + obsdate + '; nodding')
            out_filenames = run_nod(target_files, comp_date_dir, nod_offaxis, sfnum, out_flux_dir, existing_frames, mypool=mypool, show_plot=show_plot)
        else:
            print(comp_name + ' has already been extracted.')
            _out_filenames = glob(os.path.join(out_flux_dir, "*.fits"))
            out_filenames = [p.replace(out_flux_dir+'/', '') for p in _out_filenames]
        orig_night_df = add_spec_column(orig_night_df, out_filenames, these_frames, out_flux_dir)

# overwrite the df, since we added columns
orig_night_df.to_csv(df_path, index=False)