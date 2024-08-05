import os

import pandas as pd
import numpy as np
import dataset as ds
import mne
import seaborn as sns
import matplotlib.pyplot as plt
import glob

mne.set_log_level('CRITICAL')

sns.set_theme(
    context='notebook',
    style='ticks',
    palette='deep',
    font='Arial',
    font_scale=1,
    color_codes=True,
    rc={'axes.unicode_minus': False}
)


def filter_epoches(epochs, seg_duration, min_duration=10):
    bi_sign = np.zeros(epochs.selection.size + 1)
    bi_sign[1:-1] = (np.diff(epochs.selection) == 1).astype('int')
    si = epochs.selection[np.diff(bi_sign) == 1]
    ei = epochs.selection[np.diff(bi_sign) == -1]

    epoch_id_selection = []
    for sii, eii in zip(si, ei):
        if (eii - sii + 1) * seg_duration >= min_duration:
            sii_in_epoches = np.argwhere(epochs.selection == sii)[0, 0]
            eii_in_epoches = np.argwhere(epochs.selection == eii)[0, 0]
            epoch_id_selection += list(range(sii_in_epoches, eii_in_epoches + 1))

    epoch_id_selection = np.array(epoch_id_selection)
    epochs = epochs[epoch_id_selection]

    return epochs


df_info = ds.load_info()

# make session table
files = glob.glob(os.path.join(ds.path['tmp'], '**', 'raw.edf'), recursive=True)
df_sessions = pd.DataFrame()
for file in files:
    animal, session = np.array(file.split('\\'))[[-3, -2]]
    group = df_info.loc[animal, 'Genotype']

    fname = os.path.join(ds.path['tmp'], animal, session, 'raw.edf')
    raw = mne.io.read_raw_edf(fname)

    # session table
    session_length = raw.times[-1]
    df_sessions_tmp = pd.DataFrame({
        'animal_id': animal,
        'genotype': group,
        'session': session,
        'session_length': session_length
    }, index=[0])
    df_sessions = pd.concat([df_sessions, df_sessions_tmp])

df_sessions = df_sessions.reset_index(drop=True)
pd.concat([df_sessions.head(2), df_sessions.tail(2)])

# prep annot
path = os.path.join(ds.path['tmp'], 'prep', 'annot_over_loco')

seg_duration = 2
animals = df_sessions['animal_id'].unique()
df_power = pd.DataFrame()

for animal in animals:
    sessions = df_sessions.query('animal_id == @animal')['session'].unique()
    genotype = df_sessions.query('animal_id == @animal')['genotype'].iloc[0]

    for session in sessions:
        fname = os.path.join(ds.path['tmp'], animal, session, 'raw.edf')
        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)

        raw.set_channel_types({'Loco': 'bio'})
        raw.notch_filter(50)

        epochs = mne.make_fixed_length_epochs(raw.copy().filter(0.1, 40), duration=seg_duration, preload=True, id=0)
        epochs.drop_bad(flat=dict(eeg=5e-6), reject={'bio': 5e-6, 'eeg': 1000e-6})

        epochs = filter_epoches(epochs, seg_duration, min_duration=10)

        # Check session quality
        if epochs.selection.size * seg_duration >= 15 * 60:
            epochs.save(os.path.join(path, f'{animal}_{session}_epochs.fif'), overwrite=True)
            df_sessions.loc[df_sessions.query('animal_id == @animal and session == @session').index, 'reserve'] = 1
        else:
            df_sessions.loc[df_sessions.query('animal_id == @animal and session == @session').index, 'reserve'] = 0

df_sessions.to_excel(os.path.join(ds.path['tbl'], 'sessions.xlsx'))
