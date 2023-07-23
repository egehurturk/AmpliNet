import librosa, librosa.display
import matplotlib .pyplot as plt
import numpy as np
from pathlib import Path

dir_path = Path(__file__).parent.absolute()

file =  str(dir_path) + "/blues.00000.wav"
n_fft = 2048 
hop_length = 512
n_mfcc = 13

def load_file(file_name):
    signal, sr = librosa.load(file, sr=22050) # signal = 1d np array, sr = sample rate
    return signal, sr

def plot_wv(signal, sr):
    librosa.display.waveshow(signal, sr=sr)
    plt.xlabel("Time")
    plt.xlabel("Amplitude")
    plt.show()


def fft(signal):
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft) # contribution of each frequency on signal
    freq = np.linspace(0, sr, len(magnitude))

    left_freq = freq[:int(len(freq)/2)]
    left_magnitude = magnitude[:int(len(freq)/2)]
    return left_freq, left_magnitude

def plot_fft(left_freq, left_magnitude):
    plt.xlabel("Freq")
    plt.ylabel("Magnitude")
    plt.plot(left_freq, left_magnitude)
    plt.show()

def stft(signal):
    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectogram = np.abs(stft)
    return stft, spectogram

def log_spectogram(spec):
    log = librosa.amplitude_to_db(spec)
    return log

def mfcc(signal):
    mfccs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    return mfccs

def plot_spectogram(spectogram, sr, xt, yt):
    librosa.display.specshow(spectogram, sr=sr, hop_length=hop_length)
    plt.xlabel(xt)
    plt.ylabel(yt)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    signal, sr = load_file(file)
    left_freq, left_mag = fft(signal)
    spec = stft(signal)[1]
    log_spec = log_spectogram(spec)
    mfccs = mfcc(signal)
    plot_spectogram(log_spec, sr, "Time", "Frequency")
    plot_spectogram(mfccs, sr, "Time", "MFCC")