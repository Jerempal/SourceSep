import os
import numpy as np
import torch
import pyloudnorm as pyln
from torch.utils.data import Dataset
from data.utils import pad_audio_center
from config import DATASET_PERCUSSION_PATH, DATASET_MIX_AUDIO_PATH

class PreComputedMixtureDataset(Dataset):
    def __init__(self, metadata_file, sample_rate=7812):
        self.metadata = metadata_file
        self.sample_rate = sample_rate

    def _load_audio(self, file_path):
        audio = pad_audio_center(file_path)
        return audio

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load mixture
        mix_path = os.path.join(DATASET_MIX_AUDIO_PATH, row['mix_file'])
        mix_audio = self._load_audio(mix_path)

        # Load percussion
        percussion_path = os.path.join(
            DATASET_PERCUSSION_PATH, row['percussion_file'])
        percussion_audio = self._load_audio(percussion_path)
        # normalize loudness
        percussion_audio = normalize_loudness(percussion_audio)
        # scale perc with respective k (when creating mix we scale the perc with k so to compare correctly we need to scale it back)
        percussion_audio = row['k'] * percussion_audio 

        noise_combined_path = os.path.join(DATASET_MIX_AUDIO_PATH, row['noise_file']) # noise file is the name of the combined noise
        noise_audio_combined = self._load_audio(noise_combined_path)
        
        # tensorize
        mix_audio = torch.tensor(mix_audio, dtype=torch.float32)
        noise_audio_combined = torch.tensor(noise_audio_combined, dtype=torch.float32)
        # channel dim
        mix_audio = mix_audio.unsqueeze(0)
        noise_audio_combined = noise_audio_combined.unsqueeze(0)
        percussion_audio = percussion_audio.unsqueeze(0)

        return {
            'mixture_audio': mix_audio,  # audio
            'percussion_audio': percussion_audio,  # audio
            'noise_audio': noise_audio_combined,  # Combined noise audio
            'noise_classes': row['noise_classes'],  # Noise classes
            'k': row['k'],  # Mixing coefficient # coef k
            'perc_name': row['percussion_file'],  # Percussion file name
            'mix_name': row['mix_file'],  # Mixture file name
            'noise_name': row['noise_files'],  # Noise file names in the mix
            'noise_file': row['noise_file']  # Noise file name
        }


def normalize_loudness(audio, target_loudness=-3):
    meter = pyln.Meter(7812)
    loudness = meter.integrated_loudness(audio)
    if loudness == -float('inf'):
        audio /= np.max(np.abs(audio))
        loudness = meter.integrated_loudness(audio)

    audio = pyln.normalize.loudness(audio, loudness, target_loudness)

    # Ensure audio is within [-1, 1]
    max_amplitude = max(abs(audio))
    audio = audio / max_amplitude
    return torch.tensor(audio, dtype=torch.float32)
