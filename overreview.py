import os.path

import matplotlib.pyplot as plt
import mne.io
import pandas as pd
import yasa
from yasa import stft_power
import dataset as ds
import numpy as np

mne.set_log_level('CRITICAL')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rc('axes', unicode_minus=False)

from lspopt import spectrogram_lspopt


def spectrogram(raw, picks=0, f_range=(.5, 30), window=30, overlap=0.5):
    fmin = f_range[0]
    fmax = f_range[1]

    eeg = raw.get_data(picks=picks, units='uV').squeeze()
    sf = raw.info['sfreq']

    # Calculate multi-taper spectrogram
    nperseg = int(window * sf)
    noverlap = overlap * nperseg
    f, t, Sxx = spectrogram_lspopt(eeg, sf, nperseg=nperseg, noverlap=noverlap)
    Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]
    t /= 60  # Convert t to min

    return t, f, Sxx

path = os.path.join(ds.path['fig'], 'overview')

df_sessions = pd.read_excel(os.path.join(ds.path['tbl'], 'sessions.xlsx'))
animals = df_sessions['animal_id'].unique()

for animal in animals:
    sessions = df_sessions.query('animal_id == @animal')['session'].unique()
    genotype = df_sessions.query('animal_id == @animal')['genotype'].iloc[0]

    for session in sessions:
        plt.figure(figsize=(16, 3))

        # raw data
        fname = os.path.join(ds.path['tmp'], animal, session, 'raw.edf')
        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        raw.filter(0.1, 40)
        raw.notch_filter(50)

        # prep data
        fname = os.path.join(ds.path['tmp'], 'prep', 'annot_over_loco', f'{animal}_{session}_epochs.fif')
        epochs = mne.read_epochs(fname)
        epoch_length = epochs.tmax + 1/epochs.info['sfreq']



        t = raw.times / 60
        eeg = raw.get_data(picks=0, units='uV').squeeze()
        acg = raw.get_data(picks=1, units='uV').squeeze()
        sf = raw.info['sfreq']

        # Loco
        plt.subplot2grid((3, 100), (0, 0), colspan=65)
        plt.plot(t, acg, color='tab:grey', linewidth=.1)
        plt.hlines(5, t[0], t[-1], colors='tab:red')
        plt.ylim(0.1, 30)
        plt.ylabel('Loco')
        plt.title(f'{animal} | {session} | {genotype}')

        # Shadow of epoch selection
        start_row =epoch_length * epochs.selection
        end_row =epoch_length * (epochs.selection+1)
        mesh_x = np.vstack([start_row, end_row]).transpose([1, 0]).reshape(-1)
        mesh_x /= 60
        mesh_y = np.array([0, 5])
        mesh_c = np.vstack([np.ones_like(start_row), np.zeros_like(start_row)]).transpose([1, 0]).reshape(1, -1)
        mesh_c = mesh_c[:, :-1]
        plt.pcolormesh(mesh_x, mesh_y, mesh_c, cmap='Reds', alpha=.5)

        # Spectrogram
        t_p, f, p = spectrogram(raw)
        plt.xticks(ticks=[])
        plt.xlim(t_p[[0, -1]])

        plt.subplot2grid((3, 100), (1, 0), colspan=65, rowspan=2)
        plt.pcolormesh(t_p, f, p, shading='gouraud', cmap='jet', vmin=-15, vmax=30)
        plt.xlim(t_p[[0, -1]])
        plt.ylabel('Freq. (Hz)')
        plt.xlabel('Time (min)')
        del p

        # EEG
        plt.twinx()
        plt.plot(t, eeg, color='k', linewidth=.1)
        plt.ylim(-800, 400)
        plt.yticks(ticks=range(-300, 301, 200))
        plt.ylabel('EEG (μV)', y=0.66)

        # PSD
        plt.subplot2grid((3, 100), (0, 75), colspan=28, rowspan=3)
        p, f = plt.psd(eeg, Fs=sf)
        plt.yticks(ticks=range(0, 40, 5))
        plt.ylim((0, 40))
        plt.xlim((0, 30))
        plt.ylabel('PSD (dB/Hz)')
        plt.xlabel('Freq. (Hz)')

        plt.savefig(os.path.join(path, f'{animal}_{session}.png'), bbox_inches='tight')
        plt.clf()
        plt.close()
        del raw, p, eeg, acg, t
