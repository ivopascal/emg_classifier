from typing import Optional

import numpy as np
import scipy
from mne_realtime import LSLClient, MockLSLStream
import mne

DATA_FOLDER = "./data/"
BUFFER_TIME = 3  # seconds of data kept
SFREQ = 2048  # It's important to get this correct!
N_CHANNELS = 4
EPOCH_TIME = 0.2
N_CHANNELS_TO_IGNORE = 1


def fifo_buffer(buffer: np.ndarray, new_samples: np.ndarray):
    n_new_samples = new_samples.shape[1]

    buffer[:, :-n_new_samples] = buffer[:, n_new_samples:]
    buffer[:, -n_new_samples:] = new_samples


class EMGStream:
    # The data_buffer always has new data appended to the end.
    # The old data gets removed from the start.
    # This is controlled by fifo_buffer() and get_epoch_if_read()

    def __init__(self, host: Optional[str] = None, mock_file: Optional[str] = None):
        if not host:
            assert mock_file
            raw = mne.io.read_raw_gdf(DATA_FOLDER + mock_file,
                                      preload=True)
            print(f"WARNING: Using a FAKE stream from {mock_file}")
            host = "mock_stream"
            mock_stream = MockLSLStream(host, raw, ch_type='eeg', status=True)
            client = LSLClient(info=raw.info, host=host, wait_max=5)
            mock_stream.start()
        else:
            client = LSLClient(host=host, wait_max=5)

        client.start()
        self.client = client
        self.data_iterator = self.client.iter_raw_buffers()

        # We start with the first 4 seconds being 0, but that should get flushed out after 4 seconds
        self.data_buffer = np.empty(shape=(N_CHANNELS, int(BUFFER_TIME * SFREQ)))
        self.unfetched_size = 0

        self.filters = [
            mne.filter.create_filter(self.data_buffer, l_freq=30, h_freq=500, method='iir',
                                     phase='forward', sfreq=SFREQ),
            mne.filter.create_filter(self.data_buffer, l_freq=51, h_freq=49, method='iir',
                                     phase='forward', sfreq=SFREQ),
        ]

        # zi is what the state of the filter is called (rolling average). We have 2 filters.
        self.zis = [
            scipy.signal.sosfilt_zi(self.filters[0]['sos']).reshape([4, 1, 2]).repeat(repeats=N_CHANNELS, axis=1),
            scipy.signal.sosfilt_zi(self.filters[1]['sos']).reshape([4, 1, 2]).repeat(repeats=N_CHANNELS, axis=1),
        ]

    def _update_buffer(self):
        new_samples = next(self.data_iterator)
        n_new_samples = new_samples.shape[1]
        if n_new_samples == 0:
            return None

        new_samples = new_samples[N_CHANNELS_TO_IGNORE:N_CHANNELS_TO_IGNORE + N_CHANNELS, :]

        new_samples, self.zis[0] = scipy.signal.sosfilt(self.filters[0]['sos'], new_samples, zi=self.zis[0], axis=1)
        new_samples, self.zis[1] = scipy.signal.sosfilt(self.filters[0]['sos'], new_samples, zi=self.zis[1], axis=1)

        fifo_buffer(self.data_buffer, new_samples)
        self.unfetched_size += n_new_samples

    def get_epoch_if_ready(self):
        self._update_buffer()
        if self.unfetched_size <= 0.2 * SFREQ:
            return None

        self.unfetched_size -= 0.2 * SFREQ  # There is a possibility that some EMG samples get used multiple times
        return self.data_buffer[int(-0.2 * SFREQ):]

    def get_features_if_ready(self):
        epoch = self.get_epoch_if_ready()
        if epoch is None:
            return None
        epoch = epoch ** 2
        return epoch.mean(axis=1)

