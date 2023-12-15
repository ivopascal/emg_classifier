# emg_classifier
A classifier that listens to a stream of EMG data through PyLSL, and publishes classifications back to that stream. It is designed for left-right movements with EMG recorded on the lower arms.

# Installation

First clone the repository. Then install the dependencies with

```commandline
pip install pipenv
pipenv install
pipenv shell
```

# Usage

After installation, the system expects a filename with a .gdf file containing EMG collected with OpenVibe.
It also expects an available stream of EEG / EMG data. 