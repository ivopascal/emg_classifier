import os
import sys

import mne.utils

from emg_classifier.emg_stream import EMGStream
from emg_classifier.prediction_stream import PredictionStream
from emg_classifier.train_model import train_model
import argparse
from dotenv import load_dotenv, find_dotenv


def predict_from_stream(model, emg_stream):
    features = emg_stream.get_features_if_ready()
    if features is None:
        return None
    prediction = model.predict_proba([features])[0]
    mapping = {
        0: -1.0,  # Left
        1: 1.0,  # Right
        2: 0.0,  # Rest
    }

    return mapping[prediction.argmax()]


def main():
    model = train_model(file_path=args.file)
    emg_stream = EMGStream(host=args.host, mock_file=args.file,
                           ignored_channels=args.ignored)
    prediction_stream = PredictionStream()

    while True:
        prediction = predict_from_stream(model, emg_stream)
        if prediction is not None:
            prediction_stream.send_prediction(prediction)


if __name__ == "__main__":
    mne.utils.set_log_level("warning")

    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser()
    parser.add_argument('--file',
                        help='Path to the data to train the model', default=os.getenv('TRAIN_FILE'))
    parser.add_argument('--host',
                        help='Host of the EEG stream to classify', default=os.getenv('HOST'))
    parser.add_argument('--ignored',
                        help='Number of leading channels to ignore. This is usually 33, sometimes 65.',
                        default=int(os.getenv('IGNORED_CHANNELS')))
    args = parser.parse_args(sys.argv[1:])



    main()
