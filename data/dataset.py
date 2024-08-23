import os
import numpy as np
import torch
import torchaudio
import pyloudnorm as pyln
import random
import numba as nb
from torch.utils.data import Dataset
from data.utils import pad_audio_center, dynamic_loudnorm
from data.config import DATASET_PERCUSSION_PATH, DATASET_NOISE_PATH

class AudioDataset(Dataset):
    def __init__(self, metadata_file, noise_classes=None, sample_rate=7812, target_loudness=-3, max_noise_class=3, random_noise=True, classify = True):
        self.metadata = metadata_file
        self.sample_rate = sample_rate
        self.target_loudness = target_loudness
        self.max_noise_class = max_noise_class
        self.noise_classes = noise_classes
        self.noise_class_list = [
            'air_conditioner',
            'car_horn',
            'children_playing',
            'dog_bark',
            'drilling',
            'engine_idling',
            'siren',
            'jackhammer'
        ]
        # Loudness meter for normalization
        self.meter = pyln.Meter(sample_rate)
        self.random_noise = random_noise  # whether to sample random noise classes
        self.classify = classify # whether to classify noise classes
        
    def __len__(self):
        return len(self.metadata)

    def _load_and_pad_audio(self, audio_path):
        return pad_audio_center(audio_path)

    def _normalize_loudness(self, audio):
        # loudness = self.meter.integrated_loudness(audio)
        # loudness = torchaudio.transforms.Loudness(7812)(audio)
        # if torchaudio.transforms.Loudness(7812)(audio) < -np.inf:
        #     print("Warning: audio is silent, skipping normalization")
        #     # audio = pyln.normalize.peak(audio, self.target_loudness)
        #     audio = torchaudio.transforms.Vol(audio, gain=self.target_loudness, gain_type='power')
        #     # loudness = self.meter.integrated_loudness(audio)
        #     # print(f"Audio was too silent, applied peak normalization to loudness {
        #     #       loudness:.2f} LUFS")
            
        # else:
        #     audio = torchaudio.transforms.Vol(audio, gain=self.target_loudness, gain_type='power')
            
            # audio = pyln.normalize.loudness(audio, loudness, self.target_loudness)
            
        audio = torchaudio.transforms.Vol(gain=self.target_loudness, gain_type='db')(audio)
        
        max_amplitude = torch.max(torch.abs(audio))
        # if max_amplitude > 1.0:
        #     audio = audio / \
        #         max_amplitude  # Scale back to [-1, 1]

        audio = audio / max_amplitude
                
        return audio

    def create_mixture(self, percussion, noise_w, k):
        # Initialize noise as a zero tensor
        noise = torch.zeros_like(percussion)
        # noise = np.zeros_like(percussion)

        # Mix each noise waveform with the coefficient (1 - k)
        for next_segment in noise_w:
            # Rescale next_segment's loudness to match the segment using dynamic loudnorm
            rescaled_next_segment = dynamic_loudnorm(
                audio=next_segment, reference=percussion)
            # Mix with coefficient (1 - k)
            noise += (1 - k) * rescaled_next_segment

        # Dynamically normalize the noise to match the loudness of the original segment
        noise = dynamic_loudnorm(audio=noise, reference=percussion)

        # Create the final mixture: add the noise to the original segment
        scaled_percussion = k * percussion
        mixture = scaled_percussion + noise

        # Declipping: Ensure that the maximum amplitude does not exceed 1
        mixture = mixture / torch.max(torch.abs(mixture))

        return mixture, scaled_percussion

    def __getitem__(self, idx):
        # Group the metadata by percussion file so you don’t sample the same percussion multiple times
        unique_percussion_file = self.metadata['percussion_file'].unique()
        percussion_file = unique_percussion_file[idx % len(
            unique_percussion_file)]  # Loop through percussion files

        # Load percussion audio
        percussion_path = os.path.join(
            DATASET_PERCUSSION_PATH, percussion_file)
        percussion_audio = self._load_and_pad_audio(percussion_path)

        # Normalize percussion audio loudness
        percussion_audio = self._normalize_loudness(percussion_audio)

        # Now you can sample noise files dynamically
        noise_waveforms = []

        if self.random_noise is True:
            # noise_classes = random.sample(self.noise_classes, k=random.randint(1, self.max_noise_class)) if self.noise_classes is not None else random.sample(
            #     self.noise_class_list, k=random.randint(1, self.max_noise_class))
            noise_l = random.sample(self.noise_classes, k=random.randint(2, self.max_noise_class)) if self.noise_classes is not None else random.sample(
                self.noise_class_list, k=random.randint(2, self.max_noise_class))
        else:
            noise_l = self.noise_classes

        # Initialize the multi-label noise tensor
        noise_labels = torch.zeros(len(self.noise_class_list))
        # noise_labels = torch.zeros(len(
        #     self.noise_class_list)) if self.noise_classes is None else torch.zeros(len(self.noise_classes))

        # for noise_class in noise_l:
        if noise_l is not None:
            for noise_class in noise_l:
                # Sample a noise file from the selected noise class
                noise_row = self.metadata[self.metadata['noise_class']
                                            == noise_class].sample().iloc[0]
                noise_path = os.path.join(DATASET_NOISE_PATH, f"fold{
                                        noise_row['fold']}", noise_row['noise_file'])
                noise_audio = self._load_and_pad_audio(noise_path)
                noise_waveforms.append(self._normalize_loudness(noise_audio))

                # Set the noise label for the current noise class
                noise_labels[self.noise_class_list.index(noise_class)] = 1.0
                # if self.noise_classes is None:
                #     noise_labels[self.noise_class_list.index(noise_class)] = 1.0
                # else:
                    # noise_labels[self.noise_classes.index(noise_class)] = 1.0 <- this is wrong could work if we only trained for a SPECIFIC set of noise classes but why bother having len 8 then...

        # Random mixing coefficient
        k = random.choice([0.5, 0.6, 0.7, 0.8, 0.9])

        # Create mixture with dynamic loudness normalization
        mixture_audio, percussion_audio = self.create_mixture(
            percussion_audio, noise_waveforms, k)

        # on peut aussi retourner les stft des audios
        # perc_stft = torch.stft(percussion_audio, n_fft=256, hop_length=64, win_length=256, window=torch.hann_window(256), return_complex=True)
        # mix_stft = torch.stft(mixture_audio, n_fft=256, hop_length=64, win_length=256, window=torch.hann_window(256), return_complex=True)

        perc_name = percussion_file
        noise_name = [row['noise_file'] for row in self.metadata[self.metadata['noise_class'].isin(noise_l)].to_dict(orient='records')]
        
        #create channel dim
        mixture_audio = mixture_audio.unsqueeze(0)
        percussion_audio = percussion_audio.unsqueeze(0)
        
        # noise name not always same size
        
        
        if self.classify is True:
            return {
                'mixture_audio': mixture_audio,
                # 'mix_name':
                # 'mix_stft': mix_stft,
                'percussion_audio': percussion_audio,
                # 'perc_name':
                # 'perc_stft': perc_stft,
                # 'noise_labels': noise_labels  # Multi-label binary vector
            }
        else:
            return {
                'mixture_audio': mixture_audio,
                # 'mix_name':
                # 'mix_stft': mix_stft,
                'percussion_audio': percussion_audio,
                'perc_name': perc_name,
                # 'noise': noise_waveforms,
                # 'noise_name': noise_name,
            }

