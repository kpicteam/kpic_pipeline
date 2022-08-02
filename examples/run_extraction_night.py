import numpy as np
import os
import multiprocessing as mp
from glob import glob
import pandas as pd
from pipeline_utils import parse_header_night, do_extract_1d, run_nod, plot_spec, make_dirs_target, rsync_files, save_bad_frames, add_spec_column
import warnings
warnings.filterwarnings("ignore")

# UT Date of observation
obsdate = input("Enter UT Date (e.g.20220723) >>> ")
obsdate = obsdate.strip()
# print(obsdate)

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

# show extracted spectra? if True, extraction will pause after each target
show_plot = False  

# nspec file convention
filestr = "nspec"+obsdate[2:]+"_0{0:03d}.fits"

# Try to load the night_df, if exists
df_path = os.path.join(df_dir, obsdate+'.csv')
if os.path.exists(df_path):
    night_df = pd.read_csv(df_path)
    unique_targets =np.unique(night_df['TARGNAME'].values)
    print('Loaded existing nightly df.')
else:
    # generate a big df for all files from this night.
    unique_targets, night_df = parse_header_night(raw_datadir)
    night_df.to_csv(df_path, index=False)

if not 'BADFRAME' in night_df.columns:
    night_df = save_bad_frames(obsdate, night_df, df_path)
    print('Shape of df after moving bad frames:')
    print(night_df.shape)

# return df without badframes
night_df = night_df[night_df['BADFRAME'] == 0]
print(str(night_df.shape[0]) + ' number of files to extract for this night.')

print('Targets observed were: ')
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
    # VFN mode + off-axis, for a companion. Or condition
    off_axis_ind = np.where( (dist_sep >= 35) | (cgname == 'vortex'))[0]

    bkgd_ind = np.where(exptime < 15)[0]
    nod_ind = np.where(exptime >= 15)[0]

    # return indices of intersections between on/off axis and exptime
    # Case 1: bkgd and on-axis
    bkgd_onaxis = np.intersect1d(bkgd_ind, on_axis_ind)
    # Case 2: nod and on-axis
    nod_onaxis = np.intersect1d(nod_ind, on_axis_ind)
    # Case 3: nod and off-axis
    nod_offaxis = np.intersect1d(nod_ind, off_axis_ind)
    # Case 4: bkgd and off-axis - this is rare after 2021
    # also previous data won't have headers with all the info needed to run this pipeline
    # bkgd_offaxis = np.intersect1d(bkgd_ind, off_axis_ind)

    # print(nod_offaxis, nod_onaxis, bkgd_onaxis)

    if len(bkgd_onaxis) > 0:
        these_frames = target_files[bkgd_onaxis]
        # move the frames to raw dir
        rsync_files(these_frames, target_raw)
        out_filenames = [f.split('spec/')[1].replace('.fits', '_bkgdsub_spectra.fits') for f in these_frames]

        if len(glob(os.path.join(out_flux_dir, "*.fits"))) == 0:
            print('Extracting on-axis flux for ' + target_name + ' from ' + obsdate + '; bkgd sub')
            # extract!
            spectral_dataset, trace_dat = do_extract_1d(these_frames, out_flux_dir, out_filenames, use_nod_sub=False, mypool=mypool)
            filenums = [f[-9:].split('.fits')[0] for f in these_frames]
            plot_spec(trace_dat, out_filenames, out_flux_dir, filenums, target_date_dir, show_plot=show_plot)

        else:
            print(target_name + ' has already been extracted.')
        night_df = add_spec_column(night_df, out_filenames, these_frames, out_flux_dir)

    if len(nod_onaxis) > 0:
        these_frames = target_files[nod_onaxis]
        rsync_files(these_frames, target_raw)

        if len(glob(os.path.join(out_flux_dir, "*.fits"))) == 0:
            print('Extracting on-axis flux for ' + target_name + ' from ' + obsdate + '; nodding')
            out_filenames = run_nod(target_files, target_date_dir, nod_onaxis, sfnum, out_flux_dir, mypool=mypool, show_plot=show_plot)
        else:
            print(target_name + ' has already been extracted.')
            _out_filenames = glob(os.path.join(out_flux_dir, "*.fits"))
            out_filenames = [p.replace(out_flux_dir+'/', '') for p in _out_filenames]

        night_df = add_spec_column(night_df, out_filenames, these_frames, out_flux_dir)

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

        if len(glob(os.path.join(out_flux_dir, "*.fits"))) == 0:
            print('Extracting off-axis flux for ' + comp_name + ' from ' + obsdate + '; nodding')
            out_filenames = run_nod(target_files, comp_date_dir, nod_offaxis, sfnum, out_flux_dir, mypool=mypool, show_plot=show_plot)
        else:
            print(comp_name + ' has already been extracted.')
            _out_filenames = glob(os.path.join(out_flux_dir, "*.fits"))
            out_filenames = [p.replace(out_flux_dir+'/', '') for p in _out_filenames]
        night_df = add_spec_column(night_df, out_filenames, these_frames, out_flux_dir)

# overwrite the df, since we added columns
# this will not have bad frames 
night_df.to_csv(df_path, index=False)