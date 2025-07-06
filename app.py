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

# Low SHIFT_LENGTH value because higher frequency EEG data is being recorded
SHIFT_LENGTH = 0.1

# Time to record and respond to the neural signal
RUNTIME_SECONDS = 45

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

def record_live():

    # Receive data stream from Muse
    inlet = connect_eeg()

    # Get the stream info
    info = inlet.info()
    fs = int(info.nominal_srate())

    collective_eeg_data = np.empty((0, 5))

    # Set up interactive plotting
    plt.ion()
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 6))

    # Initialize empty line objects for updating later
    time_lines = [ax_time.plot([], [])[0] for _ in range(5)]
    freq_lines = [ax_freq.plot([], [])[0] for _ in range(5)]

    ax_time.set_title("Time Domain EEG Data")
    ax_time.set_xlabel("Sample")
    ax_time.set_ylabel("Amplitude (µV)")

    ax_freq.set_title("Frequency Domain EEG Data")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Magnitude")

        # Add line labels
    for i, line in enumerate(time_lines):
        line.set_label(f"Channel {i+1}")
    for i, line in enumerate(freq_lines):
        line.set_label(f"Channel {i+1}")

    # Add legends
    ax_time.legend(loc='upper right')
    ax_freq.legend(loc='upper right')

    start = time.time()
    now_time = time.time()
    while now_time < start + RUNTIME_SECONDS:
        eeg_data, timestamp = inlet.pull_chunk(
            timeout=1, max_samples=int(SHIFT_LENGTH * fs))
        np_eeg_data = np.asarray(eeg_data)

        if (np_eeg_data.shape[0] != 0):
            collective_eeg_data = np.append(collective_eeg_data, np_eeg_data, axis = 0)

            time_arr, abs = filter_data(collective_eeg_data, fs)
            freq_arr_2d, mag_arr_2d = multidimensional_tTf(collective_eeg_data, fs)

            interactive_plot(ax_time, ax_freq, time_lines, freq_lines, time_arr, freq_arr_2d, mag_arr_2d)

        now_time = time.time()
    
    # Save final figure
    plt.ioff()
    plt.savefig(f"time_&_freq_domain{now_time}.png")
    plt.show()

def interactive_plot(ax_time, ax_freq, time_lines, freq_lines, time_arr, freq_arr_2d, mag_arr_2d):
    # Update time domain lines
    for i, line in enumerate(time_lines):
        line.set_data(np.arange(time_arr.shape[0]), time_arr[:, i])
        ax_time.set_xlim(0, time_arr.shape[0])
        ax_time.set_ylim(np.min(time_arr), np.max(time_arr))

    # Update frequency domain lines
    for i, line in enumerate(freq_lines):
        line.set_data(freq_arr_2d[i], mag_arr_2d[i])
        ax_freq.set_xlim(0, np.max(freq_arr_2d[i]))
        ax_freq.set_ylim(0, 1.1)

    # Redraw
    plt.pause(0.01)
    
def multidimensional_tTf(collective_eeg_data, fs):
    num_signals, signal_length = collective_eeg_data.T.shape
    fft_length = signal_length // 2 + 1
    
    freq_arr_2d = np.zeros((num_signals, fft_length))
    mag_arr_2d = np.zeros((num_signals, fft_length))
    transposed_collective_eeg_data = collective_eeg_data.T
    for i in range(transposed_collective_eeg_data.shape[0]):
        f, m = time_to_frequency_domain(transposed_collective_eeg_data[i], fs)
        freq_arr_2d[i] = f
        mag_arr_2d[i] = m

    return freq_arr_2d, mag_arr_2d

def filter_data(collective_eeg_data, fs):
    # The filtered epoch is the epoch data with non-blink frequencies removed
    transposed_data = collective_eeg_data.T
    tr_bandpassed_data = np.zeros_like(transposed_data)
    for i in range(transposed_data.shape[0]):
        tr_bandpassed_data[i] = bandpass(transposed_data[i], fs)
    bandpassed_data = tr_bandpassed_data.T
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