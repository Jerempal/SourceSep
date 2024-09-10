#%%
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.dataset import AudioDataset
from data.utils import pad_audio_center
from data.config import DATASET_MIX_AUDIO_PATH, DATASET_PERCUSSION_PATH, DATASET_NOISE_PATH
import pyloudnorm as pyln
import matplotlib.pyplot as plt
import random

import multiprocessing as mp
mp.set_start_method('spawn', force=True)


# Load metadata
metadata = pd.read_csv(os.path.join(
    DATASET_MIX_AUDIO_PATH, "metadata.csv"))

# dataset = AudioDataset()
# dataset = AudioMixtureDatasetWithMetadata(
#     metadata_file=metadata, max_noise_classes=3, noise_classes=None)

# dataset = AudioDataset(
#     metadata_file=metadata, noise_classes=['siren', 'dog_bark'], random_noise=False)

dataset = AudioDataset(
    metadata_file=metadata, max_noise_class=2, random_noise=True)

train_data, test_data = torch.utils.data.random_split(
    dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
train_data, val_data = torch.utils.data.random_split(train_data, [int(
    0.8 * len(train_data)), len(train_data) - int(0.8 * len(train_data))])

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True, prefetch_factor=2)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True, prefetch_factor=2)

#%%
# test the dataset
# take 10 of train at random

sample = random.sample(range(len(train_data)), 10)
for i in range(10):
    data = train_data[sample[i]]
    mixture_audio = data['mixture_audio']
    percussion_audio = data['percussion_audio']
    noise_label = data['noise_labels']
    
    
    plt.plot(mixture_audio)
    plt.title(f"Mixture {i} with noise classes {noise_label}")
    plt.show()

    plt.plot(percussion_audio)
    plt.title(f"Percussion {i}")
    plt.show()

    # for j, noise in enumerate(data['noise audio']):
    #     plt.plot(noise)
    #     plt.title(f"Noise {j}")
    #     plt.show()

        # save the audio
        # import soundfile as sf
        # sf.write(f"audio_{i}_noise_{j}.wav", noise.numpy(), 7812)


#%%
# check from the 387 percussions files how many are too silent
silent_percussions = 0
p_files = metadata['percussion_file'].unique()
nb_perc = len(p_files)
for p_file in p_files:
    percussion_path = os.path.join(DATASET_PERCUSSION_PATH, p_file)
    percussion_audio = pad_audio_center(percussion_path)

    # check the loudness
    loudness = pyln.Meter(7812).integrated_loudness(percussion_audio)
    if loudness == -np.inf:
        silent_percussions += 1

print(f"Number of silent percussions: {silent_percussions} out of {
      nb_perc} ({silent_percussions/nb_perc*100:.2f}%)")
#%%
# same for the noise files
silent_noises = 0
nb_noises = metadata['noise_file'].nunique()

# get unique noise files and their fold
noises = metadata[['fold', 'noise_file']].drop_duplicates()

for i in range(nb_noises):
    noise_path = os.path.join(DATASET_NOISE_PATH, f"fold{
                              noises['fold'].iloc[i]}", noises['noise_file'].iloc[i])
    noise_audio = pad_audio_center(noise_path)

    # check the loudness
    loudness = pyln.Meter(7812).integrated_loudness(noise_audio)
    if loudness == -np.inf:
        silent_noises += 1


print(f"Number of silent noises: {silent_noises} out of {
      nb_noises} ({silent_noises/nb_noises*100:.2f}%)")

#%%
for i in range(nb_noises):
    noise_path = os.path.join(DATASET_NOISE_PATH, f"fold{
                              noises['fold'].iloc[i]}", noises['noise_file'].iloc[i])
    noise_audio = pad_audio_center(noise_path)
    loudness = pyln.Meter(7812).integrated_loudness(noise_audio)
    if loudness == -np.inf:
        print(f"noise {noises['noise_file'].iloc[i]} is too silent")
        import matplotlib.pyplot as plt
        plt.plot(noise_audio)
        plt.title(f"noise {noises['noise_file'].iloc[i]}")
        plt.show()

        print('path of the silent noise:', noise_path)

        # normalize the audio and plot it
        # noise_audio = noise_audio * 10 ** ( 60 / 20.0)
        noise_audio_peak = noise_audio / np.max(np.abs(noise_audio))

        # rescale using peak
        # noise_audio_peak = pyln.normalize.peak(noise_audio, 0)
        plt.plot(noise_audio_peak)
        plt.title(f"noise {noises['noise_file'].iloc[i]} rescaled using peak")
        plt.show()

        print('new loudness:', pyln.Meter(
            7812).integrated_loudness(noise_audio_peak))


# %%
