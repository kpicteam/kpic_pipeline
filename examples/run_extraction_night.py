import numpy as np
import os
import multiprocessing as mp
from glob import glob
import pandas as pd
import getopt
import sys
from pipeline_utils import parse_header_night, run_bkgd, run_nod, make_dirs_target, make_comp_dir, rsync_files, save_bad_frames, add_spec_column, read_user_options, get_on_off_axis
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
overwrite_files = False

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
if len(glob( os.path.join(raw_datadir, "*.fits")) ) == 0:
    raw_datadir = os.path.join("/scr3/kpic/Data/", obsdate[:4], obsdate[2:], "spec")
if len(glob( os.path.join(raw_datadir, "*.fits")) ) == 0:
    print('No raw files identified.')
    exit()

# Adjust to your machine
mypool = mp.Pool(16)
#----------------------------------------------------------------

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

    exptime = night_df.loc[night_df['TARGNAME'] == target_name, 'TRUITIME'].values
    sfnum = night_df.loc[night_df['TARGNAME'] == target_name, 'SFNUM'].values

    # get on axis and off axis files
    on_axis_ind, off_axis_ind = get_on_off_axis(night_df, target_name)

    # first check if we used more than 1 SF. If not, must do background subtraction
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
    bkgd_offaxis = np.intersect1d(bkgd_ind, off_axis_ind)

    print(nod_offaxis, nod_onaxis, bkgd_onaxis, bkgd_offaxis)

    if len(bkgd_onaxis) > 0:
        these_frames = target_files[bkgd_onaxis]
        # move the frames to raw dir
        rsync_files(these_frames, target_raw)
        if overwrite_files:
            existing_frames = []
        else:
            existing_frames = np.sort(glob(os.path.join(out_flux_dir, "*.fits")))
        # if len(existing_frames) == 0:
        if len(existing_frames) < len(these_frames):
            print('Extracting on-axis flux for ' + target_name + ' from ' + obsdate + '; bkgd sub')
            out_filenames = run_bkgd(these_frames, target_date_dir, out_flux_dir, existing_frames, mypool=mypool, show_plot=show_plot)
        else:
            print(target_name + ' has already been extracted.')
            _out_filenames = np.sort(glob(os.path.join(out_flux_dir, "*.fits")))
            out_filenames = [p.replace(out_flux_dir+'/', '') for p in _out_filenames]
            
        assert len(out_filenames) == len(these_frames)
        orig_night_df = add_spec_column(orig_night_df, out_filenames, these_frames, out_flux_dir)

    # off axis, bkgd sub
    if len(bkgd_offaxis) > 0:
        comp_date_dir, out_flux_dir, target_raw, comp_name = make_comp_dir(target_name, kpicdir, obsdate)
        these_frames = target_files[bkgd_offaxis]
        # move the frames to raw dir
        rsync_files(these_frames, target_raw)
        if overwrite_files:
            existing_frames = []
        else:
            existing_frames = np.sort(glob(os.path.join(out_flux_dir, "*.fits")))
        if len(existing_frames) < len(these_frames):
            print('Extracting off-axis flux for ' + comp_name + ' from ' + obsdate + '; bkgd sub')
            out_filenames = run_bkgd(these_frames, comp_date_dir, out_flux_dir, existing_frames, mypool=mypool, show_plot=show_plot)
        else:
            print(comp_name + ' has already been extracted.')
            _out_filenames = np.sort(glob(os.path.join(out_flux_dir, "*.fits")))
            out_filenames = [p.replace(out_flux_dir+'/', '') for p in _out_filenames]
        
        assert len(out_filenames) == len(these_frames)
        orig_night_df = add_spec_column(orig_night_df, out_filenames, these_frames, out_flux_dir)

    if len(nod_onaxis) > 0:
        these_frames = target_files[nod_onaxis]
        rsync_files(these_frames, target_raw)
        if overwrite_files:
            existing_frames = []
        else:
            existing_frames =np.sort(glob(os.path.join(out_flux_dir, "*.fits")))
        # print(len(existing_frames), len(these_frames))
        if len(existing_frames) < len(these_frames):
            print('Extracting on-axis flux for ' + target_name + ' from ' + obsdate + '; nodding')
            out_filenames = run_nod(target_files, target_date_dir, nod_onaxis, sfnum, out_flux_dir, existing_frames, mypool=mypool, show_plot=show_plot)
        else:
            print(target_name + ' has already been extracted.')
            _out_filenames = np.sort(glob(os.path.join(out_flux_dir, "*.fits")))
            out_filenames = [p.replace(out_flux_dir+'/', '') for p in _out_filenames]
        
        assert len(out_filenames) == len(these_frames)
        orig_night_df = add_spec_column(orig_night_df, out_filenames, these_frames, out_flux_dir)

    # for companion, append a letter to directory names.
    if len(nod_offaxis) > 0:
        comp_date_dir, out_flux_dir, target_raw, comp_name = make_comp_dir(target_name, kpicdir, obsdate)
        these_frames = target_files[nod_offaxis]
        rsync_files(these_frames, target_raw)
        if overwrite_files:
            existing_frames = []
        else:
            existing_frames = np.sort(glob(os.path.join(out_flux_dir, "*.fits")))

        if len(existing_frames) < len(these_frames):
            print('Extracting off-axis flux for ' + comp_name + ' from ' + obsdate + '; nodding')
            out_filenames = run_nod(target_files, comp_date_dir, nod_offaxis, sfnum, out_flux_dir, existing_frames, mypool=mypool, show_plot=show_plot)
        else:
            print(comp_name + ' has already been extracted.')
            _out_filenames =np.sort(glob(os.path.join(out_flux_dir, "*.fits")))
            out_filenames = [p.replace(out_flux_dir+'/', '') for p in _out_filenames]

        # these must be equal, otherwise the spec files won't match up
        assert len(out_filenames) == len(these_frames)
        orig_night_df = add_spec_column(orig_night_df, out_filenames, these_frames, out_flux_dir)

# overwrite the df, since we added columns
orig_night_df.to_csv(df_path, index=False)