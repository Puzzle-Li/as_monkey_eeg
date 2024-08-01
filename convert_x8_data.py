import datetime
import glob
import os.path

import mne.io
import numpy as np

import dataset as ds

mne.set_log_level('CRITICAL')


def acc_to_locomotion(acc):
    loco = np.linalg.norm(np.concatenate([np.zeros((3, 1)), np.diff(acc)], axis=1), axis=0).reshape(1, -1)
    return loco


def convert_data(path_in, path_out):
    file = glob.glob(os.path.join(path_in, '**merge.edf'), recursive=True)[0]
    raw = mne.io.read_raw_edf(file, preload=True)
    loco = acc_to_locomotion(raw.get_data()[-3:, :]) / 1e6
    raw = raw.pick(['EEG0', 'ACC0'])
    raw._data[-1, :] = loco
    raw.rename_channels({'EEG0': 'P4-M1', 'ACC0': 'Loco'})
    raw.set_channel_types({'P4-M1': 'eeg', 'Loco': 'bio'})

    raw.resample(250)
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    raw.crop(5 * 60, None)
    raw.set_meas_date(raw.info['meas_date'] + datetime.timedelta(seconds=5 * 60))
    raw.export(os.path.join(path_out, 'raw.edf'), overwrite=True)


if __name__ == "__main__":
    path_data = ds.path['raw']
    animals = os.listdir(path_data)

    for animal in animals:
        sessions = os.listdir(os.path.join(path_data, animal))

        for session in sessions:

            path_session = os.path.join(path_data, animal, session)
            path_session_out = path_session.replace(ds.path['raw'], ds.path['tmp'])
            convert_data(path_session, path_session_out)
