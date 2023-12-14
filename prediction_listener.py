import numpy as np
from pylsl import StreamInlet, resolve_stream


def main():
    print("looking for a prediction stream...")
    streams = resolve_stream('type', "Markers")
    inlet = StreamInlet(streams[0])
    print("Found stream!")
    print(streams)
    inlet.open_stream()
    predictions = np.empty((2160 * 4, 1), dtype='float32')
    for i in range(2160 * 4):
        sample, timestamp = inlet.pull_sample()
        print("got %s at time %s" % (sample, timestamp))
        predictions[i, :] = sample

    print(predictions)
    np.savetxt('data/predictions.csv', predictions, delimiter=",")


if __name__ == '__main__':
    main()