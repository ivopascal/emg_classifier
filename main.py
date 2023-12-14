from emg_classifier.emg_stream import EMGStream
from emg_classifier.prediction_stream import PredictionStream
from emg_classifier.train_model import train_model

DATA_FILE = "record-[2023.12.04-14.29.07].gdf"


def predict_from_stream(model, emg_stream):
    features = emg_stream.get_features_if_ready()
    if features is None:
        return None
    # print(features)
    prediction = model.predict_proba([features])[0]
    mapping = {
        0: -1.0,  # Left
        1: 1.0,  # Right
        2: 0.0,  # Rest
    }

    # print(prediction)
    return prediction


def main():
    model = train_model(filename=DATA_FILE)
    emg_stream = EMGStream(host=None, mock_file=DATA_FILE)
    prediction_stream = PredictionStream()

    while True:
        prediction = predict_from_stream(model, emg_stream)
        if prediction is not None:
            prediction_stream.send_prediction(prediction)


if __name__ == "__main__":
    main()
