import datetime
import glob
import os.path

import mne.io
import numpy as np
from transdata2edf import data_translate_edf_path
import dataset as ds

mne.set_log_level('CRITICAL')


def acc_to_locomotion(acc):
    loco = np.linalg.norm(np.concatenate([np.zeros((3, 1)), np.diff(acc)], axis=1), axis=0).reshape(1, -1)
    return loco


def convert_acg_data(raw):
    # Converge acg data
    loco = acc_to_locomotion(raw.get_data()[-3:, :]) / 1e6
    raw = raw.pick(['EEG0', 'ACC0'])
    raw._data[-1, :] = loco
    raw.rename_channels({'EEG0': 'P4-M1', 'ACC0': 'Loco'})
    raw.set_channel_types({'P4-M1': 'eeg', 'Loco': 'bio'})

    raw.resample(250)
    return raw


if __name__ == "__main__":
    path_data = ds.path['raw']
    files = glob.glob(os.path.join(path_data, '**', '**.eeg'), recursive=True)

    for file in files:
        animal, session = np.array(file.split('\\'))[[-3, -2]]

        path_session = os.path.split(file)[0]
        data_translate_edf_path(path_session)
        fname = glob.glob(os.path.join(path_session, f'{session}.edf'), recursive=True)[0]
        raw = mne.io.read_raw_edf(fname, preload=True)

        path_session_out = path_session.replace(ds.path['raw'], ds.path['tmp'])
        raw = convert_acg_data(raw)

        if not os.path.exists(path_session_out):
            os.makedirs(path_session_out)

        raw.export(os.path.join(path_session_out, 'raw.edf'), overwrite=True)
