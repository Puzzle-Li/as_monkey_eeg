import numpy as np
import mne
import glob
import os
import datetime
import dataset as ds


def find_over_thr_segs(data, thr, pad=0, good_min=0):
    def _find_terminals(bi_data):
        bi_data[0] = 0
        bi_data[-1] = 0
        bi_bad = np.diff(bi_data)
        bad_start_idx = np.argwhere(bi_bad == 1).reshape(-1)
        bad_end_idx = np.argwhere(bi_bad == -1).reshape(-1) + 1

        return bad_start_idx, bad_end_idx

    flag = np.abs(data) > thr * np.ones(shape=data.shape)
    bi = np.zeros(shape=data.shape)
    bi[flag] = 1

    start_idx, end_idx = _find_terminals(bi)

    if pad > 0:
        start_idx = start_idx - pad
        end_idx = end_idx + pad

        start_idx[start_idx < 0] = 0
        end_idx[end_idx > data.size] = data.size

        bi = np.zeros(shape=data.shape)
        for (si, ei) in zip(start_idx, end_idx):
            bi[int(si):int(ei)] = 1

        start_idx, end_idx = _find_terminals(bi)

    # merge bad segs
    start_idx_merged = []
    end_idx_merged = []
    if good_min > 0:
        start_idx_merged.append(start_idx[0])
        for i in range(len(start_idx)-1):
            if start_idx[i+1] - end_idx[i] >= good_min:
                end_idx_merged.append(end_idx[i])
                start_idx_merged.append(start_idx[i+1])
        end_idx_merged.append(end_idx[-1])

        start_idx = start_idx_merged
        end_idx = end_idx_merged

        bi = np.zeros(shape=data.shape)
        for (si, ei) in zip(start_idx, end_idx):
            bi[int(si):int(ei)] = 1

        start_idx, end_idx = _find_terminals(bi)
    return start_idx, end_idx, bi


def art_annot(raw):
    sf = raw.info['sfreq']

    # muscle activity
    annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
        raw,
        ch_type='eeg',
        threshold=2,
        min_length_good=0.5,
        filter_freq=[110, 120],
    )

    min_dur_muscle = 0.5  # seconds
    annot_muscle = annot_muscle[annot_muscle.duration > min_dur_muscle]

    # acg
    acg = raw.get_data(picks=1, units='uV').squeeze()
    bad_start_idx, bad_end_idx, bi_bad = find_over_thr_segs(acg, thr=5, pad=sf * 1)

    onsets = bad_start_idx / sf
    durations = (bad_end_idx - bad_start_idx) / sf
    descriptions = ['BAD_loco'] * len(bad_start_idx)
    annot_loco = mne.Annotations(
        onsets, durations, descriptions, orig_time=raw.info["meas_date"]
    )

    # eeg
    eeg = raw.get_data(picks=0, units='uV').squeeze()
    bad_start_idx, bad_end_idx, bi_bad = find_over_thr_segs(eeg, thr=1200, pad=sf * 2)

    onsets = bad_start_idx / sf
    durations = (bad_end_idx - bad_start_idx) / sf
    descriptions = ['BAD_eeg'] * len(bad_start_idx)
    annot_eeg = mne.Annotations(
        onsets, durations, descriptions, orig_time=raw.info["meas_date"]
    )

    raw.set_annotations(raw.annotations + annot_loco + annot_muscle + annot_eeg)

    return raw


def annot_to_mask(raw):
    sf = raw.info['sfreq']
    start_idx = (raw.annotations.onset * sf).astype('int')
    end_idx = ((raw.annotations.onset + raw.annotations.duration) * sf).astype('int')
    flag_ignore = np.zeros_like(raw.times, dtype='bool')
    for (si, ei) in zip(start_idx, end_idx):
        flag_ignore[si:ei + 1] = True

    return flag_ignore


if __name__ == "__main__":
    animals = os.listdir(ds.path['tmp'])

    for animal in animals:
        sessions = os.listdir(os.path.join(ds.path['tmp'], animal))

        for session in sessions:
            fname = os.path.join(ds.path['tmp'], animal, session, 'raw.edf')
            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)

            raw.set_channel_types({'Loco': 'bio'})
            raw.notch_filter(50)

            raw = art_annot(raw)
            raw.export(raw.filenames[0].replace('.edf', '_art_annot.edf'), overwrite=True)
