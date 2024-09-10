#%%
import os
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from data.utils import get_stft
import pandas as pd

output_dir_audio = 'results\\audio'
output_dir_spectrogram = 'results\\spectrogram'
residual_audio_dir = 'results\\new_residual_audio'
residual_spectrogram_dir = 'results\\new_residual_spectrogram'

# Load npy files
percussion_files = np.load('results/percussion_files.npy')
noise_names = np.load('results/noise_names.npy')
mix_names = np.load('results/mix_names.npy')
k_values = np.load('results/k_values.npy')
classes = np.load('results/classes.npy')
noise_audio = np.load('results/noise_audio.npy')
output_waveform = np.load('results/output_waveform.npy')
true_percussion = np.load('results/true_percussion.npy')
mixtures = np.load('results/mixtures.npy')
residual_audio = np.load('results/new_residual_audio.npy')

# abs path
abs_path = os.path.abspath(os.path.dirname(__file__))

# join all the paths
output_dir_audio = os.path.join(abs_path, output_dir_audio)
residual_audio_dir = os.path.join(abs_path, residual_audio_dir)

output_dir_spectrogram = os.path.join(abs_path, output_dir_spectrogram)
residual_spectrogram_dir = os.path.join(abs_path, residual_spectrogram_dir)

# or list dir
residual_files = sorted(os.listdir(residual_audio_dir), key=lambda x: int(x.split('residual_')[1].split('_')[0]))

# load metadata_inference
metadata_inference = pd.read_csv(os.path.join(abs_path, 'results\\metadata_inference.csv'))

#%%
def all_plot(mix, noise, true_perc, output_wave, residual):
    plt.figure(figsize=(15, 10))
    plt.subplot(5, 2, 1)
    plt.plot(noise[0])
    plt.title('Noise file :' + f'{metadata_inference["noise_file"][i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    print(f'File path: {os.path.join(output_dir_audio, str(metadata_inference["noise_file"][i]))}')
    
    plt.subplot(5, 2, 2)
    stft = get_stft(noise)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Noise with Class : {}'.format(metadata_inference["noise_class"][i]))

    plt.subplot(5, 2, 3)
    plt.plot(true_perc[0])
    plt.title('True percussion File: ' +
              f'{metadata_inference["true_perc_file"][i]} with value k = {metadata_inference["k_value"][i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    print(f'File path: {os.path.join(output_dir_audio, str(metadata_inference["true_perc_file"][i]))}')

    plt.subplot(5, 2, 4)
    stft = get_stft(true_perc)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('True percussion')

    plt.subplot(5, 2, 5)
    plt.plot(mix[0])
    # plt.title(f'{mix_names[i]}')
    plt.title('Mix file :' + f'{metadata_inference["mix_file"][i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    print(f'File path: {os.path.join(output_dir_audio, str(metadata_inference["mix_file"][i]))}')
          
    plt.subplot(5, 2, 6)
    stft = get_stft(mix)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mixture')

    plt.subplot(5, 2, 7)
    plt.plot(output_wave[0])
    plt.title('Separated percussion File: ' + f'{metadata_inference["separated_perc_file"][i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    # print(f'File path: Separated percussion {
        #   os.path.join(output_dir_audio, percussion_files[i])}')
    print(f'File path: {os.path.join(output_dir_audio, str(metadata_inference["separated_perc_file"][i]))}')
    
    
    plt.subplot(5, 2, 8)
    stft = get_stft(output_wave)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Separated percussion')

    plt.subplot(5, 2, 9)
    plt.plot(residual[0])
    plt.title(f'Residual file : {residual_files[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    print(f'File path: {os.path.join(
        residual_audio_dir, residual_files[i])}')
    plt.subplot(5, 2, 10)
    stft = get_stft(residual)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Residual')

    plt.tight_layout()
    plt.show()

# %%


k = np.random.choice(len(percussion_files), 5, replace=False)
for i in k:
    all_plot(mixtures[i], noise_audio[i], true_percussion[i],
             output_waveform[i], residual_audio[i])


# %%
# plot the noise only waveform of a jackhammer and it's spectrogram and a siren 

noise_jackhammer = noise_audio[metadata_inference[metadata_inference['noise_class'] == 'jackhammer'].index[14]]
noise_siren = noise_audio[metadata_inference[metadata_inference['noise_class'] == 'siren'].index[36]]

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(noise_jackhammer[0])
plt.title('Bruit:' + f'{metadata_inference["noise_file"][metadata_inference[metadata_inference["noise_class"] == "jackhammer"].index[0]]}')
plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

plt.subplot(2, 2, 2)
stft = get_stft(noise_jackhammer)
mag = torch.abs(stft)
librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
plt.colorbar(format='%+2.0f dB')
plt.title('Classe de bruit : {}'.format(metadata_inference["noise_class"][metadata_inference[metadata_inference["noise_class"] == "jackhammer"].index[0]]))

plt.subplot(2, 2, 3)
plt.plot(noise_siren[0])
plt.title('Bruit:' + f'{metadata_inference["noise_file"][metadata_inference[metadata_inference["noise_class"] == "siren"].index[0]]}')
plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1


plt.subplot(2, 2, 4)
stft = get_stft(noise_siren)
mag = torch.abs(stft)
librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
plt.colorbar(format='%+2.0f dB')
plt.title('Classe de bruit : {}'.format(metadata_inference["noise_class"][metadata_inference[metadata_inference["noise_class"] == "siren"].index[0]]))

plt.tight_layout()
plt.show()


# %%
