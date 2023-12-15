import sys

import mne
import numpy as np
import scipy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from settings import LEFT_HAND_EVENT, RIGHT_HAND_EVENT, END_OF_TRIAL_EVENT, EPOCH_TIME, MOVE_TIME, \
    CUE_TIME, EVENT_IDS


def train_model(file_path: str):
    raw = mne.io.read_raw_gdf(file_path,
                              preload=True)
    raw = raw.drop_channels("Channel 1").pick(["EX 1", "EX 2", "EX 3", "EX 4"])
    raw = raw.set_eeg_reference()
    raw = raw.set_channel_types(dict.fromkeys(raw.ch_names, "emg"))
    events, _ = mne.events_from_annotations(raw, event_id={LEFT_HAND_EVENT: 1,
                                                           RIGHT_HAND_EVENT: 2,
                                                           END_OF_TRIAL_EVENT: 3})

    filters = [
        mne.filter.create_filter(raw.get_data(), l_freq=30, h_freq=500, method='iir',
                                 phase='forward', sfreq=raw.info['sfreq']),
        mne.filter.create_filter(raw.get_data(), l_freq=51, h_freq=49, method='iir',
                                 phase='forward', sfreq=raw.info['sfreq']),
    ]

    raw_data = scipy.signal.sosfilt(filters[0]['sos'],  raw.get_data())
    raw_data = scipy.signal.sosfilt(filters[1]['sos'],  raw_data)
    raw = mne.io.RawArray(raw_data, raw.info)

    if EPOCH_TIME > MOVE_TIME:
        return
    lost_time = MOVE_TIME - EPOCH_TIME

    cue_time = CUE_TIME + lost_time / 2  # + because we want to move forward in time
    move_time = MOVE_TIME - lost_time

    epochs = mne.Epochs(
        raw,
        events,
        EVENT_IDS,
        cue_time,
        cue_time + move_time,
        baseline=None,
        preload=True,
    )

    # 80x4x411
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1] - 1
    X = np.abs(X)
    X = X.mean(axis=2)
    model = LinearDiscriminantAnalysis()
    accuracy = cross_val_score(model, X, y, cv=10).mean()
    print(f"Accuracy: {accuracy}")
    assert accuracy > 0.8
    model.fit(X, y)

    return model
