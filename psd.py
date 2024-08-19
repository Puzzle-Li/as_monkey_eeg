import os.path

import matplotlib.pyplot as plt
import mne.io
import pandas as pd
import yasa
from yasa import stft_power
import dataset as ds
import numpy as np
import seaborn as sns
from lspopt import spectrogram_lspopt

mne.set_log_level('CRITICAL')

df_sessions = pd.read_excel(os.path.join(ds.path['tbl'], 'sessions.xlsx'))
animals = df_sessions['animal_id'].unique()
df_psd = pd.DataFrame()

for animal in animals:

    sessions = df_sessions.query('animal_id == @animal')['session'].unique()
    genotype = df_sessions.query('animal_id == @animal')['genotype'].iloc[0]

    for session in sessions:
        session_id = df_sessions.query('animal_id == @animal and session == @session').index[0]

        fname = os.path.join(ds.path['tmp'], 'prep', 'annot_over_loco', f'{animal}_{session}_raw_crop_annot.edf')
        raw = mne.io.read_raw_edf(fname, preload=True)
        raw.notch_filter(50)
        raw.filter(.1, 45)
        if len(raw.annotations) > 0:
            # prep data
            for i, annot in enumerate(raw.annotations):
                df_psd_seg = raw.compute_psd(
                    picks=[0],
                    fmin=1, fmax=40,
                    tmin=annot['onset'], tmax=annot['onset'] + annot['duration'],
                    method='welch', n_fft=int(2 * raw.info['sfreq'])
                ).to_data_frame()

                df_psd_seg.insert(loc=0, column='duration', value=annot['duration'])
                df_psd_seg.insert(loc=0, column='onset', value=annot['onset'])
                df_psd_seg.insert(loc=0, column='seg_id', value=i)
                df_psd_seg.insert(loc=0, column='session', value=session)
                df_psd_seg.insert(loc=0, column='session_id', value=session_id)
                df_psd_seg.insert(loc=0, column='genotype', value=genotype)
                df_psd_seg.insert(loc=0, column='animal', value=animal)

                df_psd = pd.concat([df_psd, df_psd_seg], axis=0)

df_psd.reset_index(drop=True)
df_psd['channel'] = raw.ch_names[0]
df_psd = df_psd.rename(columns={raw.ch_names[0]: 'power'})
df_psd.to_csv(os.path.join(ds.path['tbl'], 'psd.csv'))
