import numpy as np


def microvolts_to_volts(emg: np.ndarray) -> np.ndarray:
    return emg / 1000000
