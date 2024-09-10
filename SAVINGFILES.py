
import os
import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
from data.utils import get_stft

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

# Create output directories
# output_dir_audio = 'results/audio'
# output_dir_spectrogram = 'results/spectrogram'
# os.makedirs(output_dir_audio, exist_ok=True)
# os.makedirs(output_dir_spectrogram, exist_ok=True)

# Initialize the metadata inference list
metadata_inference = []

# def save_audio_and_spectrogram(i):
#     mix_name = mix_names[i].split('/')[-1]
#     noise_name = noise_names[i].split('/')[-1]
#     perc_name = percussion_files[i]
#     k_value = k_values[i]
#     noise_class = classes[i]

#     # Get the audio signals
#     mix_audio = mixtures[i, 0]
#     noise_audio_cb = noise_audio[i, 0]
#     true_perc_audio = true_percussion[i, 0]
#     output_wave_audio = output_waveform[i, 0]

#     # Save audio files
#     sf.write(f'{output_dir_audio}/mix_{i}_{mix_name}', mix_audio, 7812)
#     sf.write(f'{output_dir_audio}/noise_{i}_{noise_name}', noise_audio_cb, 7812)
#     sf.write(f'{output_dir_audio}/true_perc_{i}_{perc_name}', true_perc_audio, 7812)
#     sf.write(f'{output_dir_audio}/separated_{i}_{perc_name}', output_wave_audio, 7812)

#     # Save spectrograms
#     save_spectrogram(mix_audio, f'mixture_{i}_{mix_name}', output_dir_spectrogram)
#     save_spectrogram(noise_audio_cb, f'noise_{i}_{noise_name}', output_dir_spectrogram)
#     save_spectrogram(true_perc_audio, f'true_percussion_{i}_{perc_name}', output_dir_spectrogram)
#     save_spectrogram(output_wave_audio, f'separated_perc_{i}_{perc_name}', output_dir_spectrogram)

#     # Append metadata information
#     metadata_inference.append({
#         'mix_file': f'mix_{i}_{mix_name}',
#         'noise_file': f'noise_{i}_{noise_name}',
#         'true_perc_file': f'true_perc_{i}_{perc_name}',
#         'separated_perc_file': f'separated_{i}_{perc_name}',
#         'k_value': k_value,
#         'noise_class': noise_class
#     })

def save_spectrogram(audio, title, output_dir):
    stft = get_stft(audio)
    mag = torch.abs(stft).cpu().numpy()
    librosa.display.specshow(librosa.amplitude_to_db(mag, ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{title}.png')
    plt.close()

if __name__ == '__main__':
    # Run in parallel
    # with Pool(cpu_count()) as pool:
    #     pool.map(save_audio_and_spectrogram, range(len(percussion_files)))

    # Save residual audio (difference between mixtures and predicted)
    residual_audio = mixtures - output_waveform
    # residual_audio_dir = 'results/residual_audio'
    # residual_spectrogram_dir = 'results/residual_spectrogram'
    residual_audio_dir = 'results/new_residual_audio'
    residual_spectrogram_dir = 'results/new_residual_spectrogram'
    os.makedirs(residual_audio_dir, exist_ok=True)
    os.makedirs(residual_spectrogram_dir, exist_ok=True)
    # save res audio npy
    np.save('results/new_residual_audio.npy', residual_audio)

    for i in tqdm(range(len(percussion_files))):
        perc_name = percussion_files[i]
        residual_audio_cb = residual_audio[i, 0]
        
        # Save residual audio
        sf.write(f'{residual_audio_dir}/residual_{i}_{perc_name}', residual_audio_cb, 7812)

        # Save residual spectrogram
        save_spectrogram(residual_audio_cb, f'residual_{i}_{perc_name}', residual_spectrogram_dir)
    
# save the metadata as a csv file outside the function
# for i in range(len(percussion_files)):
#     mix_name = mix_names[i].split('/')[-1]
#     noise_name = noise_names[i].split('/')[-1]
#     perc_name = percussion_files[i]
#     k_value = k_values[i]
#     noise_class = classes[i]

#     metadata_inference.append({
#         'mix_file': f'mix_{i}_{mix_name}',
#         'noise_file': f'noise_{i}_{noise_name}',
#         'true_perc_file': f'true_perc_{i}_{perc_name}',
#         'separated_perc_file': f'separated_{i}_{perc_name}',
#         'k_value': k_value,
#         'noise_class': noise_class
#     })
    
# metadata_df = pd.DataFrame(metadata_inference)
# metadata_df.to_csv('results/metadata_inference.csv', index=False)
# print('Metadata inference saved to results/metadata_inference.csv')

# #%%

# # exampple de l'impact des n_fft et hop_length sur la qualité de la spectrogramme
# # ainsi que la représentation de le spectrogramme de magnitude et power

# k = np.random.randint(0, len(mixtures))
# # on charge un percusision
# audio = mixtures[k,0]
# # on calcule le stft
# def get_stft1(audio):
#     stft = torch.stft(torch.tensor(audio, dtype=torch.float32).to("cuda"), n_fft=256, hop_length=64,
#                       win_length=256, window=torch.hann_window(window_length=256, device='cuda'), return_complex=True)
#     return stft

# def get_stft2(audio):
#     stft = torch.stft(torch.tensor(audio, dtype=torch.float32).to("cuda"), n_fft=1024, hop_length=256,
#                       win_length=1024, window=torch.hann_window(window_length=1024, device='cuda'), return_complex=True)
#     return stft

# stft1 = get_stft1(audio)
# stft2 = get_stft2(audio)

# # on calcule la magnitude
# mag1 = torch.abs(stft1).cpu().numpy()
# mag2 = torch.abs(stft2).cpu().numpy()

# # on affiche les spectrogrammes
# plt.figure(figsize=(15, 10))
# plt.subplot(2, 2, 1)
# librosa.display.specshow(librosa.amplitude_to_db(mag1, ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
# plt.colorbar(format='%+2.0f dB')
# plt.title('n_fft=256, hop_length=64')

# plt.subplot(2, 2, 2)
# librosa.display.specshow(librosa.amplitude_to_db(mag2, ref=np.max), y_axis='linear', x_axis='time', sr=7812,  n_fft=1024, hop_length=256)
# plt.colorbar(format='%+2.0f dB')    
# plt.title('n_fft=1024, hop_length=256')

# plt.tight_layout()
# plt.show()

# # mask example

# mask = torch.zeros_like(stft1)
# mag = torch.abs(stft1)

# # randomly select a mask 0 to 1
# mask = torch.randint(0, 2, size=mask.shape).to("cuda") # randint (0, 2) => 0 or 1

# # apply the mask
# masked_stft = mask * mag * torch.exp(1j * torch.angle(stft1))
# masked_stft.to("cpu")

# # plot the masked spectrogram
# plt.figure(figsize=(15, 10))
# plt.subplot(2, 1, 1)
# librosa.display.specshow(librosa.amplitude_to_db(mag.cpu().numpy(), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
# plt.colorbar(format='%+2.0f dB')
# plt.title('Original Spectrogram')

# plt.subplot(2, 1, 2)
# librosa.display.specshow(librosa.amplitude_to_db(torch.abs(masked_stft).cpu().numpy(), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
# plt.colorbar(format='%+2.0f dB')
# plt.title('Masked Spectrogram')

# plt.tight_layout()
# plt.show()


# # %%
