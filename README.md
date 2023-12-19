# emg_classifier
A classifier that listens to a stream of EMG data through PyLSL, and publishes classifications back to that stream. It is designed for left-right movements with EMG recorded on the lower arms.

# Installation

## Windows without Python
First install Python using the Microsoft store.
Then, run `pip install pipenv` to install Pipenv. 
After this, `pipenv` should be added to the path. Where pipenv has been installed will be shown in the output during installation.
Restart `cmd` and check that pipenv is available by running `pipenv`

## Installation after Python

First clone the repository or download as a zip if git is not installed. 
Then install the dependencies with:
```commandline
pip install pipenv
pipenv install
pipenv shell
```

# Usage

After installation, the system expects a filename with a .gdf file containing EMG collected with OpenVibe.
It also expects an available stream of EEG / EMG data. 

## Usage protocol in the lab

 1. Turn on the Biosemi amplified with the battery 
 2. Run OpenVibe and the Acquisition Server
 3. Select the motor_imagery_acquisiton_Ivo.xdf file in the OpenVibe designer
 4. Change the path in the GDF File Writer to a location you can access
 5. Set up the acquisition server to collect 1 EEG channel. Also set the preferences to allow drift correction. 
 5. Run a recording with OpenVibe to collect EMG data. This will take about 2.5 minutes
 6. Stop the OpenVibe acquisition server
 7. Run biosemi.exe (search in windows menu)
 8. Set the subset in biosemi.exe to 32. Don't worry too much about the other settings. Click LINK to start
 9. Run LabRecorder.exe to see the EEG stream from Biosemi. It will look something like `Biosemi (SOME_CODE)`
10. Copy `.env.template` to `.env`, there set the file path and set `HOST=SOME_CODE`
11. Run `python main.py`
12. Select both the EEG stream and the Prediction stream in LabRecorder and click start.


