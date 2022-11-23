import numpy as np
import os
import multiprocessing as mp
from glob import glob
import matplotlib.pyplot as plt
import kpicdrp.data as data
import sys
import pandas as pd
from pipeline_utils import get_throughput, make_dirs_target, add_thru_starmag, query_starmag, get_on_off_axis
import warnings
warnings.filterwarnings("ignore")

## Script to calculate throughput for all files for a night, add append to df.

obsdate = input("Enter UT Date (e.g.20220723) >>> ")
obsdate = obsdate.strip()
# print(obsdate)
datelist = [obsdate]
# datalist = ['20220806', '20220723', '20220722', '20220721', '20220720']
for obsdate in datelist:
    # what to plot
    plot_x_var = 'el'  #'dar'
    # main kpic dir
    kpicdir = "/scr3/kpic/KPIC_Campaign/" # main data dir
    show_plot = False  # show extracted spectra
    filestr = "nspec"+obsdate[2:]+"_0{0:03d}.fits"

    # where to save throughput data
    thrudir = os.path.join(kpicdir, 'nightly_tables', 'throughput_data', obsdate)
    if not os.path.exists(thrudir):
        os.makedirs(thrudir)

    # load the night_df, with bad frames column
    df_path = kpicdir+'nightly_tables/'+obsdate+'.csv'
    night_df = pd.read_csv(df_path)

    # filter out bad frames
    night_df = night_df[night_df['BADFRAME'] == 0]
    # K band only
    night_df = night_df[night_df['FILTER'] == 'Kband-new']

    unique_targets =np.unique(night_df['TARGNAME'].values)
    print(night_df.shape)
    # print(night_df.head())

    # save this dictionary, and keep adding to it
    # kmag_dict = {'GQLup': 7.096, 'HD145647':6.033,  'HD26670':6.054, 'HIP76310': 7.172, 'HIP81497':0.44, 
    #             'Kelt-9': 7.482, 'TOI1684': 5.859, 'ups_Her': 4.880, 'HD206893': 5.593, 'HR8799': 5.240, 'HIP95771': 0.52,
    #             'lamCap': 5.568, 'RXJ1609.5-2105': 8.916, 'HD130396': 6.221, 'HIP79098': 5.707, 'HD130767': 6.781}

    # wvs solution
    wv_path = glob('/scr3/kpic/KPIC_Campaign/calibs/'+obsdate+'/wave/*_wvs.fits')[0]
    print(wv_path)
    wv_soln = data.Wavecal(filepath=wv_path)

    print(wv_soln.labels)

    # iterate over target
    all_thru = {'sf1':[], 'sf2':[], 'sf3':[], 'sf4':[] }
    all_el = {'sf1':[], 'sf2':[], 'sf3':[], 'sf4':[] }
    all_dar = {'sf1':[], 'sf2':[], 'sf3':[], 'sf4':[] }
    # all_thru = {'sf2':[], 'sf3':[], 'sf4':[] }
    # all_el = {'sf2':[], 'sf3':[], 'sf4':[] }
    # all_dar = {'sf2':[], 'sf3':[], 'sf4':[] }

    for j, target_name in enumerate(unique_targets):
        if (not target_name == 'etaPeg') and (not target_name == '79Cyg'):
            print('Onto ' + target_name)

            target_date_dir, out_flux_dir, target_raw = make_dirs_target(kpicdir, target_name, obsdate)
            
            # which files belong to this target
            target_files = night_df.loc[night_df['TARGNAME'] == target_name, 'FILEPATH'].values

            # Only compute throughput for on-axis
            dist_sep = night_df.loc[night_df['TARGNAME'] == target_name, 'FIUDSEP'].values
            exptime = night_df.loc[night_df['TARGNAME'] == target_name, 'TRUITIME'].values
            sfnum = night_df.loc[night_df['TARGNAME'] == target_name, 'SFNUM'].values
            elev = night_df.loc[night_df['TARGNAME'] == target_name, 'EL'].values
            dar = night_df.loc[night_df['TARGNAME'] == target_name, 'DAR'].values
            cgname = night_df.loc[night_df['TARGNAME'] == target_name, 'CORONAGRAPH'].values

            origname = night_df.loc[night_df['TARGNAME'] == target_name, 'TARGNAME_ORIG'].values[0]

            # using 35 mas, to handle cases where we intentionally offset RV star by 30 mas (prevent saturation...)
            # also custom and MDA (apodizer)
            # on_axis_ind = np.where( (dist_sep < 35) & ((cgname == 'pupil_mask') | (cgname == 'Custom') | (cgname == 'apodizer') | (cgname == 'pypo_out')) )[0]
            on_axis_ind, off_axis_ind = get_on_off_axis(night_df, target_name)
            # on and off axis files
            # requires df to have these values
            all_spec_files = night_df.loc[night_df['TARGNAME'] == target_name, 'SPECFILE'].values
            # print(len(all_spec_files), all_spec_files[0])

            if len(on_axis_ind) > 0:
                # take only on-axis ones
                
                all_spec_onaxis = all_spec_files[on_axis_ind]
                on_files = out_flux_dir + '/' + all_spec_onaxis

                # print(len(on_files), on_files)
                on_sf = sfnum[on_axis_ind]
                on_exptime = exptime[on_axis_ind]
                on_elev = elev[on_axis_ind]
                on_dar = dar[on_axis_ind]

                if origname == 'TOI-1684':
                    origname = 'HIP 20334'
                if origname == '2M0122-2439':
                    origname = '2MASS J01225093-2439505'
                if origname == 'HIP 213179':
                    origname = 'HD 213179'

                kmag, rpmag, gmag = query_starmag(origname)
                print(origname, kmag, rpmag, gmag)
                #kmag = kmag_dict[target_name]

                all_thru95 = []
                thru_array = []
                all_thru_file = []
                all_mags = {'2MASSK': [], 'GaiaG': [], 'GaiaRP': []}

                for i, (f, this_sf) in enumerate(zip(on_files, on_sf)):
                    # print(this_sf)
                    peak_thru, thru_all_orders, all_wvs = get_throughput(f, wv_soln, kmag, bb_temp=7000, fib='s'+this_sf[-1], show_plot=False)
                    print(peak_thru*100)
                    sf = on_sf[i]
                    el = on_elev[i]
                    dar = on_dar[i]

                    all_thru['sf'+sf[-1]].append(np.round(peak_thru, 4))
                    # elevation
                    all_el['sf'+sf[-1]].append(el)
                    # DAR
                    all_dar['sf'+sf[-1]].append(dar)
                    all_thru95.append(np.round(peak_thru, 4))

                    # save the array of throughputs and wvs
                    thru_file = os.path.join(thrudir, f.split('nspec')[1].replace('.fits', '_thru.npy'))
                    np.save(thru_file, thru_all_orders)
                    wvs_file = os.path.join(thrudir, 'wave_solution.npy')
                    np.save(wvs_file, all_wvs)

                    all_thru_file.append(thru_file)
                    all_mags['2MASSK'].append(kmag)
                    all_mags['GaiaG'].append(gmag)
                    all_mags['GaiaRP'].append(rpmag)

                # add throughput and wvs columns
                first_target = (j == 0)
                print(first_target)
                night_df = add_thru_starmag(night_df, all_spec_onaxis, all_thru95, all_thru_file, all_mags, first_target=first_target)

    # print(all_thru)
    print('Number of frames on each SF (1, 2, 3, 4)')
    print(len(all_thru['sf1']), len(all_thru['sf2']), len(all_thru['sf3']), len(all_thru['sf4']))

    # Save in the night_df, and make a new df
    fig, ax = plt.subplots(figsize=(10,6))
    for s in ['1', '2', '3', '4']:
    #for s in ['2', '3', '4']:
        if len(all_thru['sf'+s]) > 0:
            if plot_x_var == 'dar':
                x_array = np.array(all_dar['sf'+s])
            elif plot_x_var == 'el':
                x_array = np.array(all_el['sf'+s])

            ax.scatter(x_array, np.array(all_thru['sf'+s])*100, label='SF'+s)

    ax.legend(fontsize=12)
    if plot_x_var == 'dar':
        ax.set_xlabel('DAR (mas)', fontsize=12)
    elif plot_x_var == 'el':
        ax.set_xlabel('Elevation (deg)', fontsize=12)

    ax.set_ylabel('95th percentile throughput (%)', fontsize=12)
    ax.tick_params(labelsize=12)
    plt.savefig('../../kpic_phase2/throughput_plots/' + obsdate + '_thruput_'+plot_x_var+'.png', dpi=100, bbox_inches='tight')
    # plt.show()

    night_df.to_csv(df_path, index=False)



    