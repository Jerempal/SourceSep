# Helper functions (assumed to be defined elsewhere)
import os
import librosa
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np


def pad_audio_center(file_path, target_length=4 * 7812):  # Example target length
    audio, sr = librosa.load(file_path, sr=None)
    if len(audio) < target_length:
        padding = (target_length - len(audio)) // 2
        audio = np.pad(audio, (padding, target_length -
                       len(audio) - padding), 'constant')
    return audio[:target_length]


def create_mixture(percussion_audio, noise_audio, k):
    mixture_audio = k * percussion_audio + (1 - k) * noise_audio
    return mixture_audio, k


# Constants
DATASET_PERCUSSION_PATH = 'path_to_percussion_data'
DATASET_NOISE_PATH = 'path_to_noise_data'


class AudioMixtureDataset(Dataset):
    def __init__(self, metadata_file=None, k=None, noise_class=None, percussion_dir=DATASET_PERCUSSION_PATH, noise_dir=DATASET_NOISE_PATH, sample_rate=7812, n_fft=256, hop_length=64, target_loudness=-30, max_noise_sources=3):
        self.metadata = metadata_file
        self.k = k
        self.noise_class = noise_class
        self.noise_class_list = [
            'air_conditioner',
            'car_horn',
            'children_playing',
            'dog_bark',
            'drilling',
            'engine_idling',
            'siren',
            'jackhammer']

        self.percussion_dir = percussion_dir
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_loudness = target_loudness
        self.max_noise_sources = max_noise_sources

        self.percussion_files = [f for f in os.listdir(
            percussion_dir) if f.endswith('.wav')]

        self.noise_files = {}
        for fold in range(1, 11):
            self.noise_files[fold] = [f for f in os.listdir(
                os.path.join(noise_dir, f"fold{fold}")) if f.endswith('.wav')]

    def __len__(self):
        if self.metadata is not None and self.noise_class is not None:
            return len(self.metadata[self.metadata['noise_class'] == self.noise_class])
        elif self.metadata is not None:
            return len(self.metadata)
        else:
            return len(self.percussion_files)

    def __getitem__(self, idx):
        if self.metadata is not None:
            if self.noise_class is not None:
                row = self.metadata[self.metadata['noise_class']
                                    == self.noise_class].iloc[idx]
            else:
                row = self.metadata.iloc[idx]

            k = self.k if self.k is not None else np.random.uniform(0.5, 0.9)

            percussion_path = os.path.join(
                DATASET_PERCUSSION_PATH, row['percussion_file'])
            percussion_audio = pad_audio_center(percussion_path)
            percussion_audio /= np.max(np.abs(percussion_audio))

            noise_path = os.path.join(DATASET_NOISE_PATH, f"fold{
                                      row['fold']}", row['noise_file'])
            noise_audio = pad_audio_center(noise_path)
            noise_audio /= np.max(np.abs(noise_audio))

            mixture_audio, _ = create_mixture(percussion_audio, noise_audio, k)

            return {
                'percussion_audio': percussion_audio,
                'mixture_audio': mixture_audio,
                'noise_class': torch.tensor(self.noise_class_list.index(row['noise_class'])),
            }

        else:
            percussion_path = os.path.join(
                self.percussion_dir, self.percussion_files[idx])
            percussion_audio = pad_audio_center(percussion_path)
            percussion_audio /= np.max(np.abs(percussion_audio))

            num_noise_sources = np.random.randint(
                1, self.max_noise_sources + 1)
            selected_noise_files = []
            for _ in range(num_noise_sources):
                fold = np.random.randint(1, 11)
                noise_file = np.random.choice(
                    self.noise_files[fold], replace=False)
                selected_noise_files.append(
                    {'fold': fold, 'noise_file': noise_file})

            noise = np.zeros_like(percussion_audio)
            for noise_file in selected_noise_files:
                noise_path = os.path.join(self.noise_dir, f'fold{
                                          noise_file["fold"]}', noise_file["noise_file"])
                noise_src = pad_audio_center(noise_path)
                noise_src /= np.max(np.abs(noise_src))
                noise += noise_src

            noise /= np.max(np.abs(noise))

            k = np.random.uniform(0.5, 0.9)
            scaled_perc = k * percussion_audio
            scaled_noise = (1 - k) * noise
            mixture = scaled_perc + scaled_noise
            mixture /= np.max(np.abs(mixture))

            percussion_stft = librosa.stft(
                scaled_perc, n_fft=self.n_fft, hop_length=self.hop_length)
            mixture_stft = librosa.stft(
                mixture, n_fft=self.n_fft, hop_length=self.hop_length)

            return {
                'percussion_audio': scaled_perc,
                'noise_audio': scaled_noise,
                'mixture_audio': mixture,
                'percussion_stft': percussion_stft,
                'mixture_stft': mixture_stft,
            }


class AudioMixtureDataset(Dataset):
    def __init__(self, metadata_file=None, k=None, noise_class=None, percussion_dir=DATASET_PERCUSSION_PATH, noise_dir=DATASET_NOISE_PATH, sample_rate=7812, n_fft=256, hop_length=64, target_loudness=-30, max_noise_sources=3):
        self.metadata = pd.read_csv(metadata_file) if metadata_file else None
        self.k = k
        self.noise_class = noise_class
        self.noise_class_list = [
            'air_conditioner',
            'car_horn',
            'children_playing',
            'dog_bark',
            'drilling',
            'engine_idling',
            'siren',
            'jackhammer']

        self.percussion_dir = percussion_dir
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_loudness = target_loudness
        self.max_noise_sources = max_noise_sources

        self.percussion_files = [f for f in os.listdir(
            percussion_dir) if f.endswith('.wav')]

        self.noise_files = {}
        for fold in range(1, 11):
            self.noise_files[fold] = [f for f in os.listdir(
                os.path.join(noise_dir, f"fold{fold}")) if f.endswith('.wav')]

    def __len__(self):
        if self.metadata is not None and self.noise_class is not None:
            return len(self.metadata[self.metadata['noise_class'] == self.noise_class])
        elif self.metadata is not None:
            return len(self.metadata)
        else:
            return len(self.percussion_files)

    def __getitem__(self, idx):
        if self.metadata is not None:
            if self.noise_class is not None:
                row = self.metadata[self.metadata['noise_class']
                                    == self.noise_class].iloc[idx]
            else:
                row = self.metadata.iloc[idx]

            k = self.k if self.k is not None else np.random.uniform(0.5, 0.9)

            percussion_path = os.path.join(
                DATASET_PERCUSSION_PATH, row['percussion_file'])
            percussion_audio = pad_audio_center(percussion_path)
            percussion_audio /= np.max(np.abs(percussion_audio))

            noise_path = os.path.join(DATASET_NOISE_PATH, f"fold{
                                      row['fold']}", row['noise_file'])
            noise_audio = pad_audio_center(noise_path)
            noise_audio /= np.max(np.abs(noise_audio))

            mixture_audio, _ = create_mixture(percussion_audio, noise_audio, k)

            return {
                'percussion_audio': percussion_audio,
                'mixture_audio': mixture_audio,
                'noise_class': torch.tensor(self.noise_class_list.index(row['noise_class'])),
            }

        else:
            percussion_path = os.path.join(
                self.percussion_dir, self.percussion_files[idx])
            percussion_audio = pad_audio_center(percussion_path)
            percussion_audio /= np.max(np.abs(percussion_audio))

            num_noise_sources = np.random.randint(
                1, self.max_noise_sources + 1)
            selected_noise_files = []
            selected_noise_classes = []
            for _ in range(num_noise_sources):
                fold = np.random.randint(1, 11)
                noise_file = np.random.choice(
                    self.noise_files[fold], replace=False)
                selected_noise_files.append(
                    {'fold': fold, 'noise_file': noise_file})

                # Get the noise class from the file name or another mapping
                noise_class = self.get_noise_class(
                    noise_file)  # Implement this function
                selected_noise_classes.append(noise_class)

            noise = np.zeros_like(percussion_audio)
            for noise_file in selected_noise_files:
                noise_path = os.path.join(self.noise_dir, f'fold{
                                          noise_file["fold"]}', noise_file["noise_file"])
                noise_src = pad_audio_center(noise_path)
                noise_src /= np.max(np.abs(noise_src))
                noise += noise_src

            noise /= np.max(np.abs(noise))

            k = np.random.uniform(0.5, 0.9)
            scaled_perc = k * percussion_audio
            scaled_noise = (1 - k) * noise
            mixture = scaled_perc + scaled_noise
            mixture /= np.max(np.abs(mixture))

            percussion_stft = librosa.stft(
                scaled_perc, n_fft=self.n_fft, hop_length=self.hop_length)
            mixture_stft = librosa.stft(
                mixture, n_fft=self.n_fft, hop_length=self.hop_length)

            return {
                'percussion_audio': scaled_perc,
                'noise_audio': scaled_noise,
                'mixture_audio': mixture,
                'percussion_stft': percussion_stft,
                'mixture_stft': mixture_stft,
                'noise_classes': torch.tensor([self.noise_class_list.index(nc) for nc in selected_noise_classes]),
            }

    def get_noise_class(self, noise_file):
        # Implement logic to determine the noise class from the noise file name or another mapping
        # For example, if noise_file contains the noise class name
        for noise_class in self.noise_class_list:
            if noise_class in noise_file:
                return noise_class
        return 'unknown'

