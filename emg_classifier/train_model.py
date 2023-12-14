import mne
import numpy as np
import scipy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

DATA_FOLDER = './data/'
left_hand_event = "769"
right_hand_event = "770"
end_of_trial_event = "800"

event_ids = dict(left=1, right=2, rest=3)

CUE_TIME = 1.25
MOVE_TIME = 3.75
t = 0.2


def train_model(filename: str):
    raw = mne.io.read_raw_gdf(DATA_FOLDER + filename,
                              preload=True)
    raw = raw.drop_channels("Channel 1").pick(["EX 1", "EX 2", "EX 3", "EX 4"])
    raw = raw.set_eeg_reference()
    raw = raw.set_channel_types(dict.fromkeys(raw.ch_names, "emg"))
    events, _ = mne.events_from_annotations(raw, event_id={left_hand_event: 1, right_hand_event: 2, end_of_trial_event: 3})

    filters = [
        mne.filter.create_filter(raw.get_data(), l_freq=30, h_freq=500, method='iir',
                                 phase='forward', sfreq=raw.info['sfreq']),
        mne.filter.create_filter(raw.get_data(), l_freq=51, h_freq=49, method='iir',
                                 phase='forward', sfreq=raw.info['sfreq']),
    ]

    raw_data = scipy.signal.sosfilt(filters[0]['sos'],  raw.get_data())
    raw_data = scipy.signal.sosfilt(filters[1]['sos'],  raw_data)
    raw = mne.io.RawArray(raw_data, raw.info)

    if t > MOVE_TIME:
        return
    lost_time = MOVE_TIME - t

    cue_time = CUE_TIME + lost_time / 2  # + because we want to move forward in time
    move_time = MOVE_TIME - lost_time

    epochs = mne.Epochs(
        raw,
        events,
        event_ids,
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
    print(f"Accuracy: {cross_val_score(model, X, y, cv=10).mean()}")
    model.fit(X, y)
    print(model.predict(X))
    return model