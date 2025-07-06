from locale import normalize
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask
from scipy.signal import butter, lfilter, argrelextrema
import pyautogui

from utils import update_buffer, get_last_data
# Import functions from the utils module
from pylsl import StreamInlet, resolve_byprop

app = Flask(__name__)

# The length of the shift between epochs in seconds
SHIFT_LENGTH = 0.1

# Time to record and respond to the neural signal
RUNTIME_SECONDS = 5

def bandpass(data, fs, low= 0.1 , high= 5.5 ,order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return lfilter(b, a, data, axis=0)
        
"""
Attempt to connect to the Muse EEG device and throw an error if not found

Return: The StreamInlet that provides live EEG data
"""
def connect_eeg():

    # Search for active LSL streams and connect to Muse
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
    else:
        print('Found it!')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)

    return inlet

"""
Records live neural signal from Muse EEG and listens for blink actions

One blink: Right arrow key is pressed (advance slides)
Two blinks: (Within a set time window) Play/pause key is pressed
"""
def record_live():

    # Receive data stream from Muse
    inlet = connect_eeg()

    # Get the stream info
    info = inlet.info()
    fs = int(info.nominal_srate())

    collective_eeg_data = np.empty((0, 5))

    start = time.time()
    now_time = time.time()
    while now_time < start + RUNTIME_SECONDS:

        # Obtain EEG data from the LSL stream
        eeg_data, timestamp = inlet.pull_chunk(
            timeout=1, max_samples=int(SHIFT_LENGTH * fs))
        np_eeg_data = np.asarray(eeg_data)
        print(np_eeg_data)
        if (not(np_eeg_data.shape[0] == 0)):
            collective_eeg_data = np.append(collective_eeg_data, np_eeg_data, axis = 0)

        now_time = time.time()

    print(f'collective_eeg_data: {collective_eeg_data.shape}')

    time_arr, abs = filter_data(collective_eeg_data, fs)

    num_signals, signal_length = collective_eeg_data.T.shape
    fft_length = signal_length // 2 + 1
    
    freq_arr_2d = np.zeros((num_signals, fft_length))
    mag_arr_2d = np.zeros((num_signals, fft_length))
    transposed_collective_eeg_data = collective_eeg_data.T
    for i in range(transposed_collective_eeg_data.shape[0]):
        f, m = time_to_frequency_domain(transposed_collective_eeg_data[i], fs)
        freq_arr_2d[i] = f
        mag_arr_2d[i] = m

    print(f'freq_arr_2d: {freq_arr_2d.shape}')

    plt.figure(figsize=(10, 6))
    # First subplot (top)
    plt.subplot(2, 1, 1)
    labels = [f"Datastream {i+1}" for i in range(collective_eeg_data.shape[1])]
    plt.plot(time_arr, label = labels)
    plt.title("Time Domain EEG Data")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (µV)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freq_arr_2d.T, mag_arr_2d.T, label = labels)
    plt.title("Frequency Domain EEG Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.savefig(f"time_&_freq_domain{now_time}.png")


    print(collective_eeg_data)

def plot_time_and_freq(freqs, mag, entire_raw_data, whole_timestamps, now_time):
    plt.figure(figsize=(10, 6))

    # First subplot (top)
    plt.subplot(2, 1, 1)
    plt.plot(entire_raw_data)
    plt.title("Live EEG Data Epoch")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (µV)")

    # df_freq = pd.DataFrame({'frequencies': freqs, 'magnitude': mag})
    # df_freq.to_csv('freq_data_with_timestamps_{int(now_time)}.csv', index=False)

    # Second subplot (bottom)
    plt.subplot(2, 1, 2)
    plt.plot(freqs, mag)
    plt.title("Live EEG Data Epoch")
    plt.xlim([0, 50])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.savefig(f"time_freq_{int(now_time)}.png")

def export_raw_csv(blink_time_all, blink_data_all, whole_timestamps, entire_raw_data, now_time):
    df_time = pd.DataFrame({'timestamp': whole_timestamps, 'eeg_value': entire_raw_data})
    df_time.to_csv(f'time_data_with_timestamps_{int(now_time)}.csv', index=False)
    df_blink = pd.DataFrame({'timestamp': blink_time_all,'eeg_value': blink_data_all})
    df_blink.to_csv(f'blink_only_time_series_{int(now_time)}.csv', index=False)
            
def filter_data(collective_eeg_data, fs):

    # The filtered epoch is the epoch data with non-blink frequencies removed
    print(collective_eeg_data[:, 1])
    transposed_data = collective_eeg_data.T
    tr_bandpassed_data = np.zeros_like(transposed_data)
    for i in range(transposed_data.shape[0]):
        tr_bandpassed_data[i] = bandpass(transposed_data[i], fs)
    bandpassed_data = tr_bandpassed_data.T
    print(f'filtered_epoch_1: {bandpassed_data}')
    filtered_epoch_np = np.array(bandpassed_data)

    # Append the filtered epoch data to the whole data set
    new_arr = np.array([])

    if len(new_arr) == 0:
        new_arr = filtered_epoch_np
    else:
        new_arr = np.concatenate((collective_eeg_data, filtered_epoch_np))

    return new_arr, np.abs(bandpassed_data) #abs

def time_to_frequency_domain(signal, fs):
    n = len(signal)
    magnitude = np.abs(np.fft.rfft(signal)) / n
    normalized_magnitude = magnitude / np.max(magnitude)
    freqs = np.fft.rfftfreq(n, 1/fs)

    return freqs, normalized_magnitude

if __name__ == '__main__':
    record_live()