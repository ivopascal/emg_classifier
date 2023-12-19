DATA_FOLDER = "./data/"
BUFFER_TIME = 3  # seconds of data kept
SFREQ = 2048  # It's important to get this correct!
N_CHANNELS = 4
EPOCH_TIME = 0.2


LEFT_HAND_EVENT = "769"
RIGHT_HAND_EVENT = "770"
END_OF_TRIAL_EVENT = "800"  # Used for rests

EVENT_IDS = dict(left=1, right=2, rest=3)

CUE_TIME = 1.25
MOVE_TIME = 3.75

