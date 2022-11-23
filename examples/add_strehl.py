from pipeline_utils import add_strehl_data
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

main_dir = '/scr3/kpic/KPIC_Campaign/nightly_tables/'
date = '20221114'
log_dir = '/scr3/kpic/Data/'+date[2:]+'/'

log_path = log_dir + date[2:] + '_log.csv'
log_df = pd.read_csv(log_path)
df_path = main_dir + date + '.csv'
night_df = pd.read_csv(df_path)

# Only compute throughput for on-axis
dist_sep = night_df['FIUDSEP'].values
cgname = night_df['CORONAGRAPH'].values
spec_files = night_df['SPECFILE'].values
ut_times = night_df['UTTIME'].values
fiucgx = night_df['FIUCGX'].values

# using 35 mas, to handle cases where we intentionally offset RV star by 30 mas (prevent saturation...)
# also custom and MDA (apodizer)
on_axis_ind = np.where( (dist_sep < 35) & ((cgname == 'pupil_mask') | (cgname == 'Custom') | (cgname == 'apodizer') | (cgname == 'pypo_out') | (fiucgx < 5)) & (fiucgx < 5) )[0]
these_frames = spec_files[on_axis_ind]
these_times = ut_times[on_axis_ind]

add_strehl_data(night_df, these_frames, these_times, log_df)

night_df.to_csv(df_path, index=False)