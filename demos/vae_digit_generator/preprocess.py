import librosa
import numpy as np
import os
import pickle

class DataLoader:
    """
    Responsible for loading an audio file
    """
    def __init__(self, sample_rate, duration, mono):
        self.sr = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal, sr = librosa.load(file_path, sr=self.sr, duration=self.duration, mono=self.mono)
        return signal
        
class Padder:
    """
    Responsible for applying padding to array
    """
    def __init__(self, mode = "constant"):
        self.mode = mode

    def left_pad(self, array, num_missing):
        """
        Prepend the array with num_missing number of elements
        """
        padded = np.pad(array, (num_missing, 0), mode=self.mode)
        return padded

    def right_pad(self, array, num_missing):
        """
        Append the array with num_missing number of elements
        """
        padded = np.pad(array, (0, num_missing), mode=self.mode)
        return padded

class LogSpectogramExtracter:
    """
    Extract the log valued spectogram from an audio signal
    """
    
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal, n_fft = self.frame_size, hop_length=self.hop_length)[:-1]
        spect = np.abs(stft)
        log_spect = librosa.amplitude_to_db(spect)
        return log_spect

class MinMaxNormalizer:
    """
    Apply min max normalization to the array
    """

    def __init__(self, min_val, max_val) :
        self.min = min_val
        self.max = max_val
    
    def normalize(self, array):
        normal = (array - array.min()) / (array.max() - array.min())
        normal = normal * (self.max - self.min) + self.min
        return normal

    def denormalize(self, normalized_array, og_min, og_max):
        array = (normalized_array - self.min) / (self.max - self.min)
        array = array * (og_max - og_min) + og_min
        return array

class Saver:
    """
    Responsible for saving features
    """

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir,
                                 "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path

class PreprocessingPipeline:
    """
    Processes audio files in a directory
    """

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sr * loader.duration)
    
    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")

        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self.min_max_values[save_path] = {
            "min": feature.min(),
            "max": feature.max()
        }

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal


FRAME_SIZE = 512
HOP_LENGTH = 256
DURATION = 0.74  # in seconds
SAMPLE_RATE = 22050
MONO = True

SPECTROGRAMS_SAVE_DIR = "./data/spectograms"
MIN_MAX_VALUES_SAVE_DIR = "./data/min_max"
FILES_DIR = "./data/audio"


if __name__ == "__main__":

    loader = DataLoader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectogramExtracter(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormalizer(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)