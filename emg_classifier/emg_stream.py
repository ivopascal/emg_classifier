from typing import Optional

import numpy as np
import scipy
from mne_realtime import LSLClient, MockLSLStream
import mne

from settings import N_CHANNELS, BUFFER_TIME, SFREQ, MOCK_TIME_DILATION


def fifo_buffer(buffer: np.ndarray, new_samples: np.ndarray):
    n_new_samples = new_samples.shape[1]

    buffer[:, :-n_new_samples] = buffer[:, n_new_samples:]
    buffer[:, -n_new_samples:] = new_samples


class EMGStream:
    # The data_buffer always has new data appended to the end.
    # The old data gets removed from the start.
    # This is controlled by fifo_buffer() and get_epoch_if_ready()

    def __init__(self, host: Optional[str] = None, mock_file: Optional[str] = None, ignored_channels=1):
        if not host or host == "mock" or host == "None":
            assert mock_file
            raw = mne.io.read_raw_gdf(mock_file,
                                      preload=True)
            print(f"WARNING: Using a FAKE stream from {mock_file}")
            host = "mock_stream"
            mock_stream = MockLSLStream(host, raw, ch_type='eeg', status=False, time_dilation=MOCK_TIME_DILATION)
            client = LSLClient(info=raw.info, host=host, wait_max=5)
            mock_stream.start()
        else:
            client = LSLClient(info=mne.create_info(ch_names=ignored_channels+8, sfreq=SFREQ), host=host, wait_max=5)
        self.ignored_channels = ignored_channels
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

        new_samples = new_samples[self.ignored_channels:self.ignored_channels + N_CHANNELS, :]
        new_samples = new_samples - new_samples.mean(axis=0)

        new_samples, self.zis[0] = scipy.signal.sosfilt(self.filters[0]['sos'], new_samples, zi=self.zis[0], axis=1)
        new_samples, self.zis[1] = scipy.signal.sosfilt(self.filters[0]['sos'], new_samples, zi=self.zis[1], axis=1)

        fifo_buffer(self.data_buffer, new_samples)
        self.unfetched_size += n_new_samples

    def get_epoch_if_ready(self):
        self._update_buffer()
        if self.unfetched_size < 0.2 * SFREQ:
            return None

        self.unfetched_size -= 0.2 * SFREQ  # There is a possibility that some EMG samples get used multiple times
        return self.data_buffer[:, int(-0.2 * SFREQ):]

    def get_features_if_ready(self):
        epoch = self.get_epoch_if_ready()
        if epoch is None or np.isnan(epoch.sum()):
            return None
        # (9, 409)
        epoch = np.abs(epoch)
        x = epoch.mean(axis=1)

        if np.isnan(x.sum()):
            return None
        return x
