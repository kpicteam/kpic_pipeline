import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy.signal import medfilt

# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True

# Dates to include
# dates = ['20220808', '20220806', '20220723', '20220722', '20220721', '20220720', '20220718']
dates = ['20221111','20221112', '20221113', '20221114']  # just July

# change paths to match your machine
kpicdir = "/scr3/kpic/KPIC_Campaign/"  # main directory, where nightly_tables is located
outdir = '../../kpic_phase2/throughput_plots/'  # directory to output plots
c_list = ['xkcd:cerulean', 'xkcd:coral', 'xkcd:goldenrod']

# True: plot throughput curve for each frame
# False: plot 95 percentile throughput (single point) for each frame
plot_thru_file = False

# which variable to plot against throughput
plot_x_var = 'STREHL'  # Choose from data in .csv file.

# science fiber - which to plot
fig, ax = plt.subplots(figsize=(10,6))
for sf_ind, sf_num in enumerate(['2','3','4']):
    all_wave = []
    all_thru_ar = []
    all_thru95 = np.array([])
    all_filename = np.array([])

    for date_i, obsdate in enumerate(dates):
        thrudir = os.path.join(kpicdir, 'nightly_tables', 'throughput_data', obsdate)
        # load the night_df, with bad frames column
        df_path = kpicdir+'nightly_tables/'+obsdate+'.csv'
        night_df = pd.read_csv(df_path)
        # filter out bad frames
        night_df = night_df[night_df['BADFRAME'] == 0]
        # use only frames with thrufile
        print(obsdate)
        thru_df = night_df[~night_df['THRU95'].isnull()]

        # pick of SF
        thru_df = thru_df[thru_df['SFNUM'] == 'science fiber ' + sf_num]

        # omit very bad frames with throughput < 0.2%
        thru_df = thru_df[thru_df['THRU95'] > 0.001]

        thru_df = thru_df[thru_df['STREHL'] < 0.4]

        # grab all throughput files
        thru_files = thru_df['THRUFILE'].values
        thru_95 = thru_df['THRU95'].values
        all_thru95 = np.append(all_thru95, thru_95)
        all_filename = np.append(all_filename, thru_files)
        print(len(thru_95), len(all_thru95))

        x_values = thru_df[plot_x_var].values
        if plot_thru_file:
            wave = np.load(os.path.join(thrudir, 'wave_solution.npy') )[int(sf_num)-1]
            for fil in thru_files:
                thru_array = np.load(fil)
                # print(wave.shape, thru_array.shape)
                # ax.scatter(wave.flatten(), thru_array.flatten(), color='xkcd:cerulean', alpha=0.4, s=5)

                smooth_thru = np.zeros((9, 2048))
                for i, throughput_order in enumerate(thru_array):
                    smor = medfilt(throughput_order, kernel_size=11)
                    smooth_thru[i] = smor

                all_thru_ar.append(smooth_thru.flatten())
                
                # or plot 1 order
                # ax.scatter(wave[6], thru_array[6], color='xkcd:cerulean', alpha=0.4, s=5)
        else:
            # ax.scatter(x_values, thru_95*100, label=obsdate)
            if date_i == 0:
                ax.scatter(x_values, thru_95*100, label='SF'+ sf_num, color=c_list[sf_ind],alpha=0.7)
            else:
                ax.scatter(x_values, thru_95*100, label='__nolegend__', color=c_list[sf_ind],alpha=0.7)

    print('SF ' + sf_num)
    # print(all_thru95)
    median_thru95 = np.median(all_thru95)
    print('Max throughput 95%')
    max_ind = np.argmax(all_thru95)
    print(np.max(all_thru95))
    print('Median throughput 95%')
    print(median_thru95)

    # print(max_ind, all_filename[max_ind])

    ax.tick_params(labelsize=14)

    if plot_thru_file:
        print('plots median of all frame')
        # print(np.nanmedian(all_thru_ar, axis=0))
        ax.plot(wave.flatten(), np.nanmedian(all_thru_ar, axis=0), color='black', linestyle='dashdot')

        # save to numpy file
        median_thru_file = os.path.join(outdir, 'median_array.npy')
        np.save(median_thru_file, np.nanmedian(all_thru_ar, axis=0).reshape((9, 2048)) )

        ax.set_xlabel('Wavelength (microns)', fontsize=14)
        ax.set_ylabel('Throughput', fontsize=14)
        ax.legend(fontsize=10)
        plt.savefig(outdir + '/sf' + sf_num + '_' + str(len(dates))+'nights_thruput_vs_wave.png', dpi=100, bbox_inches='tight')

    else:
        # ax.axhline(median_thru95 * 100, label='Median = '+str(median_thru95 * 100) + '%', color='black', linestyle='dashdot')
        ax.set_xlabel(plot_x_var, fontsize=14)
        ax.set_ylabel('95th percentile throughput (%)', fontsize=14)
        ax.legend(fontsize=10)
        # ax.set_title('Science Fiber ' + sf_num)
        plt.savefig(outdir + '/' + '_'.join(dates) + '_thruput_vs_'+plot_x_var+'.png', dpi=100, bbox_inches='tight')

plt.show()




