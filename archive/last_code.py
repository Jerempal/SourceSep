# %%
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from data.config import *
from data.utils import save_checkpoint, load_checkpoint
from  data.dataset import PreComputedMixtureDataset
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from model.Last_model import ResUNetv2

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# mp.set_start_method('spawn', force=True)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

# %%
# Load metadata
metadata = pd.read_csv(os.path.join(
    DATASET_MIX_AUDIO_PATH, "metadata.csv"))

dataset = PreComputedMixtureDataset(metadata_file=metadata)

# %%
# Group the metadata by percussion file
grouped_metadata = metadata.groupby('percussion_file')

# Get all unique percussion files
unique_perc_files = metadata['percussion_file'].unique()

# Split based on percussion files, not individual mixtures
train_perc_files, test_perc_files = train_test_split(
    unique_perc_files, test_size=0.2, random_state=42)
train_perc_files, val_perc_files = train_test_split(
    # 0.25 of the remaining for validation
    train_perc_files, test_size=0.25, random_state=42)

# Create train, validation, and test datasets by filtering the metadata
train_metadata = metadata[metadata['percussion_file'].isin(train_perc_files)]
val_metadata = metadata[metadata['percussion_file'].isin(val_perc_files)]
test_metadata = metadata[metadata['percussion_file'].isin(test_perc_files)]

# see test metadata where the noise files are repeated with different k values
diff_k = test_metadata2.groupby('noise_files').filter(lambda x: x['k'].nunique() > 2)


# Save the indices (if needed)
train_indices = train_metadata.index.tolist()
val_indices = val_metadata.index.tolist()
test_indices = test_metadata.index.tolist()

# np.save('train_indices_new_last.npy', train_indices)
# np.save('val_indices_new_last.npy', val_indices)
# np.save('test_indices_new_last.npy', test_indices)

# Use these new indices for your DataLoader
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=32, num_workers=8, persistent_workers=True)
# val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=32, num_workers=8, persistent_workers=True)
# test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=32, num_workers=8, persistent_workers=True)

train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=32)
val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=32)
test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=32)

# based on indices see if files of percussion (by looking at the names) are repeted in the train, val and test sets

train_files = metadata.iloc[train_indices]
val_files = metadata.iloc[val_indices]
test_files = metadata.iloc[test_indices]

train_perc_files = train_files['percussion_file']
val_perc_files = val_files['percussion_file']
test_perc_files = test_files['percussion_file']

train_perc_files = train_perc_files.apply(lambda x: x.split('/')[-1])
val_perc_files = val_perc_files.apply(lambda x: x.split('/')[-1])
test_perc_files = test_perc_files.apply(lambda x: x.split('/')[-1])

train_perc_files = train_perc_files.to_list()
val_perc_files = val_perc_files.to_list()
test_perc_files = test_perc_files.to_list()

print(len(train_perc_files))
print(len(val_perc_files))
print(len(test_perc_files))

print(len(set(train_perc_files)))
print(len(set(val_perc_files)))
print(len(set(test_perc_files)))

print(len(set(train_perc_files).intersection(set(val_perc_files))))
print(len(set(train_perc_files).intersection(set(test_perc_files))))
print(len(set(val_perc_files).intersection(set(test_perc_files))))

# all the files repeat themselves IT'S BAD because it means the model will have seen all the percussion file
#%%
# print(f"The number of unique noise classes: {test_metadata['noise_classes'].nunique()}")
# print(f"The count of each k value: {test_metadata['k'].value_counts()}")
print(f"The number of unique percussion files: {
      test_metadata['percussion_file'].nunique()}")
print(f"The number of unique mix files: {test_metadata['mix_file'].nunique()}")
# print(f"The number of unique noise files: {test_metadata['noise_files'].nunique()}")
# print(f"The count of each noise file: {test_metadata['noise_files'].value_counts()}")
print(test_metadata.groupby("noise_files").size().describe())

# %%
data = next(iter(train_loader))

for i in range(5):
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.plot(data['mixture_audio'][i, 0].numpy())
    plt.title(f'Mixture {i}')
    plt.subplot(3, 1, 2)
    plt.plot(data['percussion_audio'][i, 0].numpy())
    plt.title(f'Percussion {i} with k = {data["k"][i]}')
    plt.subplot(3, 1, 3)
    plt.plot(data['noise_audio'][i, 0].numpy())
    plt.title(f'Noise {i} with k = {
              1-data["k"][i]}, noise class = {data["noise_classes"][i]}')

    # print path of mix and noise
    print(os.path.join(DATASET_MIX_AUDIO_PATH, data['mix_name'][i]))
    print(os.path.join(DATASET_MIX_AUDIO_PATH, data['noise_file'][i]))

    plt.tight_layout()
    plt.show()

# %%
# Define the model, optimizer and loss function
model = ResUNetv2(in_c=1, out_c=32).to("cuda")
optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True, fused=True)

# criterion = nn.MSELoss()
criterion = nn.L1Loss()
device = "cuda"
sdr_metric = SignalDistortionRatio().to("cuda")
si_sdr_metric = ScaleInvariantSignalDistortionRatio().to("cuda")

# %%
# Train the model

train_losses = []
val_losses = []
val_sdr = []
val_si_sdr = []
best_val_loss = np.inf
patience = 5
num_epochs = 30

start_epoch = 0
# data augmentation :


for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {
                     epoch + 1}/{num_epochs} Training Loss: {train_loss:.8f}", colour='green')
    for i, batch in enumerate(train_bar):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Move data to device
        mixture = batch['mixture_audio'].to(device)
        true_percussion = batch['percussion_audio'].to(device)

        # Forward pass
        output_waveform = model(mixture)['waveform']

        loss = criterion(output_waveform, true_percussion)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} Training Loss: {train_loss/(i+1):.8f}")

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    val_sdr_epoch = 0
    val_si_sdr_epoch = 0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss:.8f}", colour='red')
    with torch.no_grad():
        for i, batch in enumerate(val_bar):
            mixture = batch['mixture_audio'].to(device)
            true_percussion = batch['percussion_audio'].to(device)

            output_waveform = model(mixture)['waveform']

            loss = criterion(output_waveform, true_percussion)
            val_loss += loss.item()

            # Calculate SDR and SI-SDR
            si_sdr = si_sdr_metric(output_waveform, true_percussion)
            sdri = sdr_metric(true_percussion, output_waveform) - sdr_metric(true_percussion, mixture)
            val_sdr_epoch += sdri
            val_si_sdr_epoch += si_sdr
                    
            val_bar.set_description(f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss/(i+1):.8f}, SI-SDR: {val_si_sdr_epoch/(i+1):.8f}, SDR Improvement: {val_sdr_epoch/(i+1):.8f}")

        val_loss /= len(val_loader)
        val_sdr_epoch /= len(val_loader)
        val_si_sdr_epoch /= len(val_loader)

        val_losses.append(val_loss)
        val_sdr.append(val_sdr_epoch)
        val_si_sdr.append(val_si_sdr_epoch)

        # save checkpoint
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir='checkpoint',
                        filename='checkpoint_last_epoch_{}'.format(epoch))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 5
            torch.save(model.state_dict(), 'best_model_last.pth')
            print("Model improved. Saving the model...")

        else:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                break


# %%
# Plot the results
plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()


# %%
# Load the best model
model.load_state_dict(torch.load('best_model_last.pth', weights_only=True))
#%%
# load sampler
train_indices = np.load('train_indices_new_last.npy')
val_indices = np.load('val_indices_new_last.npy')
test_indices = np.load('test_indices_new_last.npy')

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# load laoder again
train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=25)
val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=25)
test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=25)


# %%
# test without all the stft, mulitlabel etc, we only keep the names of the files, value of k

model.eval()
test_loss = 0
percussion_files_l = []  # names of percussion files in each batch
noise_names_l = []       # names of noise files in each batch
mix_names_l = []         # names of mix files in each batch
k_values_l = []          # k values in each batch
classes_l = []           # noise classes in each batch
noise_audio_l = []       # noise audio in each batch

output_waveform_l = []   # output waveform in each batch
true_percussion_l = []   # true percussion in each batch
mixtures_l = []          # mixtures in each batch

si_sdr_tensor = []       # SI-SDR in each batch

# Progress bar for testing
test_bar = tqdm(test_loader, desc=f"Testing Loss: {
                test_loss:.8f}", colour='red')

with torch.no_grad():
    for i, batch in enumerate(test_bar):
        # Move data to the device
        mixture = batch['mixture_audio'].to(device)
        true_percussion = batch['percussion_audio'].to(device)

        # Extract relevant information from the batch
        percussion_files = batch['perc_name']
        noise_names = batch['noise_name']
        mix_names = batch['mix_name']
        k_values = batch['k']
        classes = batch['noise_classes']
        noise_audio = batch['noise_audio']

        # Forward pass through the model
        output_waveform = model(mixture)['waveform']

        # Calculate the loss
        loss = criterion(output_waveform, true_percussion)
        test_loss += loss.item()

        # Calculate SI-SDR for the current batch
        si_sdr = calculate_si_sdr(true_percussion, output_waveform)

        # Append values to the lists
        percussion_files_l.extend(percussion_files)
        noise_names_l.extend(noise_names)
        mix_names_l.extend(mix_names)
        k_values_l.extend(k_values)
        classes_l.extend(classes)
        noise_audio_l.extend(noise_audio)

        output_waveform_l.append(output_waveform)
        true_percussion_l.append(true_percussion)
        mixtures_l.append(mixture)

        si_sdr_tensor.append(si_sdr)

        # Update progress bar description
        test_bar.set_description(f"Testing Loss: {
                                 test_loss/(i+1):.8f}, SI-SDR: {torch.mean(torch.tensor(si_sdr_tensor)):.8f}")

    # Final loss and SI-SDR
    test_loss /= len(test_loader)
    final_si_sdr = torch.mean(torch.tensor(si_sdr_tensor))
    print(f"Testing Loss: {test_loss:.8f}, SI-SDR: {final_si_sdr:.8f}")

# %%

# change the lists to numpy arrays
percussion_files = np.array(percussion_files_l)
noise_names = np.array(noise_names_l)
mix_names = np.array(mix_names_l)
k_values = np.array(k_values_l)
classes = np.array(classes_l)

# audios to numpy
noise_audio = np.array(noise_audio_l)
output_waveform = torch.cat(output_waveform_l, dim=0).cpu().numpy()
true_percussion = torch.cat(true_percussion_l, dim=0).cpu().numpy()
mixtures = torch.cat(mixtures_l, dim=0).cpu().numpy()

# si_sdr_tensor to numpy
si_sdr_tensor = torch.tensor(si_sdr_tensor).numpy()

# %%
# save this lists
os.makedirs('results', exist_ok=True)

# all the names of the percussion files
np.save('results/percussion_files.npy', percussion_files)
# all the names of the noise files
np.save('results/noise_names.npy', noise_names)
np.save('results/mix_names.npy', mix_names)  # all the names of the mix files
np.save('results/k_values.npy', k_values)  # all the k values
np.save('results/classes.npy', classes)  # all the noise classes
np.save('results/noise_audio.npy', noise_audio)  # all the noise audio
# all the output waveform audio
np.save('results/output_waveform.npy', output_waveform)
# all the true percussion audio
np.save('results/true_percussion.npy', true_percussion)
np.save('results/mixtures.npy', mixtures)  # all the mixtures audio
np.save('results/si_sdr.npy', si_sdr_tensor)  # all the si-sdr values

# %%

# shape of all
arrays = [percussion_files, noise_names, mix_names, k_values,
          classes, noise_audio, output_waveform, true_percussion, mixtures]
for array in arrays:
    print(array.shape)

# %%

# plot the waveform of the separated percussion with their respective noise, class and k value
# we can plot the waveform of the noise as well
for i in range(5):
    plt.figure(figsize=(10, 5))
    plt.subplot(4, 1, 1)
    plt.plot(mixtures[i, 0])
    plt.title(f'Mixture {mix_names[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

    plt.subplot(4, 1, 2)
    # in the mix they are multiplied by (1-k) so we do the same here to compare
    plt.plot(noise_audio[i, 0])
    plt.title(f'Noise {noise_names[i]}, Class {
              classes[i]}, (1-k) = {1-k_values[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

    plt.subplot(4, 1, 3)
    # in the mix they are multiplied by k so we do the same here to compare
    plt.plot(true_percussion[i, 0])
    plt.title(f'True Percussion {percussion_files[i]}, k = {k_values[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

    plt.subplot(4, 1, 4)
    plt.plot(output_waveform[i, 0])
    plt.title(f'Separated Percussion {percussion_files[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

    plt.tight_layout()
    plt.show()

# %%


def get_stft(audio):
    stft = torch.stft(torch.tensor(audio, dtype=torch.float32).to("cuda"), n_fft=256, hop_length=64,
                      win_length=256, window=torch.hann_window(window_length=256, device='cuda'), return_complex=True)
    return stft

# compute their spectrogram


def plot_spectrogram_alone(audio, title):
    stft = get_stft(audio)
    mag = torch.abs(stft)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# %%
# plot the spectrogram of the separated percussion with their respective noise, class and k value
# we can plot the spectrogram of the noise as well


for i in range(5):
    plot_spectrogram_alone(mixtures[i], f'Mixture {mix_names[i]}')

    plot_spectrogram_alone(noise_audio[i], f'Noise {noise_names[i]}, Class {
                           classes[i]}, (1-k) = {1-k_values[i]}')

    plot_spectrogram_alone(true_percussion[i], f'True Percussion {
                           percussion_files[i]}, k = {k_values[i]}')

    plot_spectrogram_alone(output_waveform[i], f'Separated Percussion {
                           percussion_files[i]}')

# %%


def plot_audio_spectrogram(audio, title):
    stft = get_stft(audio)
    mag = torch.abs(stft)

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(audio[0])
    plt.title(title)
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# %%
# now plot all audios with their spectrograms, with audio at the top and spectrogram at the bottom


for i in range(5):
    plot_audio_spectrogram(mixtures[i], f'Mixture {mix_names[i]}')
    plot_audio_spectrogram(noise_audio[i], f'Noise {noise_names[i]}, Class {
                           classes[i]}, (1-k) = {1-k_values[i]}')
    plot_audio_spectrogram(true_percussion[i], f'True Percussion {
                           percussion_files[i]}, k = {k_values[i]}')
    plot_audio_spectrogram(output_waveform[i], f'Separated Percussion {
                           percussion_files[i]}')

# %%
i = -1
plt.figure(figsize=(10, 5))
plt.subplot(4, 1, 1)
plt.plot(mixtures[i, 0])
plt.title(f'Mixture {mix_names[i]}')
plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

plt.subplot(4, 1, 2)
# in the mix they are multiplied by (1-k) so we do the same here to compare
plt.plot(noise_audio[i, 0])
plt.title(f'Noise {noise_names[i]}, Class {
          classes[i]}, (1-k) = {1-k_values[i]}')
plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

plt.subplot(4, 1, 3)
# in the mix they are multiplied by k so we do the same here to compare
plt.plot(true_percussion[i, 0])
plt.title(f'True Percussion {percussion_files[i]}, k = {k_values[i]}')
plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

plt.subplot(4, 1, 4)
plt.plot(output_waveform[i, 0])
plt.title(f'Separated Percussion {percussion_files[i]}')
plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

plt.tight_layout()
plt.show()
# %%
# verify if the file : percussion_files[-1] is in the training
percussion_file_test_name = percussion_files[-1]
percussion_file_test_name = percussion_file_test_name.split('/')[-1]

train_files = metadata.iloc[train_indices]
train_perc_files = train_files['percussion_file']
train_perc_files = train_perc_files.apply(lambda x: x.split('/')[-1])
train_perc_files = train_perc_files.to_list()

# test with this file hps_2024_02_21-09_16_08
percu_file = 'hps_2024_02_21-09_16_08'
percu_file = percu_file.split('/')[-1]
try:
    train_perc_files.index(percu_file)
    print('File is in the training set')
except:
    print('File is not in the training set')

try:
    train_perc_files.index(percussion_file_test_name)
    print('File is in the training set')
except:
    print('File is not in the training set')

# now test it for all the test files
percussion_files = np.array(percussion_files)
percussion_files = np.array([perc.split('/')[-1] for perc in percussion_files])

ct = 0
for perc_file in percussion_files:
    try:
        train_perc_files.index(perc_file)
        print(f'{perc_file} is in the training set')
    except:
        ct += 1

print(f'{ct} files are not in the training set')

# %%
# plot all 4 on same figure with audio on top and spectrogram at the bottom


def all_plot(mix, noise, true_perc, output_wave, perc_title, noise_title):
    plt.figure(figsize=(15, 10))

    plt.subplot(4, 2, 1)
    plt.plot(noise[0])
    plt.title(noise_title)
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

    plt.subplot(4, 2, 2)
    stft = get_stft(noise)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Noise')

    plt.subplot(4, 2, 3)
    plt.plot(true_perc[0])
    plt.title('True Percussion ' + perc_title)
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

    plt.subplot(4, 2, 4)
    stft = get_stft(true_perc)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('True Percussion')

    plt.subplot(4, 2, 5)
    plt.plot(mix[0])
    plt.title(f'Mixture {mix_names[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

    plt.subplot(4, 2, 6)
    stft = get_stft(mix)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mixture')

    plt.subplot(4, 2, 7)
    plt.plot(output_wave[0])
    plt.title('Separated Percussion')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1

    plt.subplot(4, 2, 8)
    stft = get_stft(output_wave)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Separated Percussion')

    plt.tight_layout()
    plt.show()


# %%
for i in range(5):
    all_plot(mixtures[i], noise_audio[i], true_percussion[i], output_waveform[i], f'Percussion File: {percussion_files[i]} with value k = {
             k_values[i]}', f'Noise File: {noise_names[i]} with Class: {classes[i]} and value (1-k) = {1-k_values[i]}')

# %%

# load the files before the all plot function
# all the names of the percussion files
percussion_files = np.load('results/percussion_files.npy')
# all the names of the noise files
noise_names = np.load('results/noise_names.npy')
mix_names = np.load('results/mix_names.npy')  # all the names of the mix files
k_values = np.load('results/k_values.npy')  # all the k values
classes = np.load('results/classes.npy')  # all the noise classes
noise_audio = np.load('results/noise_audio.npy')  # all the noise audio
# all the output waveform audio
output_waveform = np.load('results/output_waveform.npy')
# all the true percussion audio
true_percussion = np.load('results/true_percussion.npy')
mixtures = np.load('results/mixtures.npy')  # all the mixtures audio
si_sdr_tensor = np.load('results/si_sdr.npy')  # all the si-sdr values


# %%
# save all the files in a folder
output_dir_audio = 'results/audio'
output_dir_spectrogram = 'results/spectrogram'

os.makedirs(output_dir_audio, exist_ok=True)
os.makedirs(output_dir_spectrogram, exist_ok=True)

for i in range(len(percussion_files)):
    mix_name = mix_names[i].split('/')[-1]
    noise_name = noise_names[i].split('/')[-1]
    perc_name = percussion_files[i]

    mix_audio = mixtures[i, 0]  # batch, channel
    noise_audio_cb = noise_audio[i, 0]  # batch, channel
    true_perc_audio = true_percussion[i, 0]  # batch, channel
    output_wave_audio = output_waveform[i, 0]  # batch, channel

    # save audio
    sf.write(f'{output_dir_audio}/{mix_name}', mix_audio, 7812)
    sf.write(f'{output_dir_audio}/{noise_name}', noise_audio_cb, 7812)
    sf.write(f'{output_dir_audio}/{perc_name}', true_perc_audio, 7812)
    sf.write(f'{output_dir_audio}/separated_{perc_name}',
             output_wave_audio, 7812)

    # save spectrogram
    mix_stft = get_stft(mix_audio)
    mix_mag = torch.abs(mix_stft)
    mix_mag = mix_mag.cpu().numpy()
    librosa.display.specshow(librosa.amplitude_to_db(
        mix_mag, ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mixture {mix_name}')
    plt.tight_layout()
    plt.savefig(f'{output_dir_spectrogram}/mixture_{mix_name}.png')
    plt.close()

    noise_stft = get_stft(noise_audio_cb)
    noise_mag = torch.abs(noise_stft)
    noise_mag = noise_mag.cpu().numpy()
    librosa.display.specshow(librosa.amplitude_to_db(
        noise_mag, ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Noise {noise_name}')
    plt.tight_layout()
    plt.savefig(f'{output_dir_spectrogram}/noise_{noise_name}.png')
    plt.close()

    perc_stft = get_stft(true_perc_audio)
    perc_mag = torch.abs(perc_stft)
    perc_mag = perc_mag.cpu().numpy()
    librosa.display.specshow(librosa.amplitude_to_db(
        perc_mag, ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'True Percussion {perc_name}')
    plt.tight_layout()
    plt.savefig(f'{output_dir_spectrogram}/percussion_{perc_name}.png')
    plt.close()

    output_wave_stft = get_stft(output_wave_audio)
    output_wave_mag = torch.abs(output_wave_stft)
    output_wave_mag = output_wave_mag.cpu().numpy()
    librosa.display.specshow(librosa.amplitude_to_db(
        output_wave_mag, ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Separated Percussion {perc_name}')
    plt.tight_layout()
    plt.savefig(f'{output_dir_spectrogram}/separated_{perc_name}.png')
    plt.close()

# %%

residual_audio = abs(mixtures - output_waveform)

output_dir_audio = 'results/residual_audio'
output_dir_spectrogram = 'results/residual_spectrogram'

os.makedirs(output_dir_audio, exist_ok=True)
os.makedirs(output_dir_spectrogram, exist_ok=True)
unique_idx = 0

for i in tqdm(range(len(percussion_files))):
    mix_name = mix_names[i].split('/')[-1]
    noise_name = noise_names[i].split('/')[-1]
    perc_name = percussion_files[i]
    residual_audio_cb = residual_audio[i, 0]  # batch, channel

    # save audio
    sf.write(f'{output_dir_audio}/residual_{perc_name}{unique_idx}{
             mix_name}{noise_name}', residual_audio_cb, 7812)

    # save spectrogram
    residual_stft = get_stft(residual_audio_cb)
    residual_mag = torch.abs(residual_stft)
    residual_mag = residual_mag.cpu().numpy()
    librosa.display.specshow(librosa.amplitude_to_db(
        residual_mag, ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Residual {perc_name}')
    plt.tight_layout()
    plt.savefig(
        f'{output_dir_spectrogram}/residual_{perc_name}{unique_idx}{mix_name}{noise_name}.png')
    plt.close()
    unique_idx += 1


# %%

# modify the all plot function to include the residual audio and spectrogram

# os absolute path
abs_path = os.path.abspath('archive_results')

output_dir_audio = 'audio\\'
# output_dir_spectrogram = 'results/spectrogram'
output_dir_res_audio = 'residual_audio\\'
output_dir_res_spectrogram = 'residual_spectrogram\\'

# join all the paths
output_dir_audio = os.path.join(abs_path, output_dir_audio)
output_dir_res_audio = os.path.join(abs_path, output_dir_res_audio)
# output_dir_spectrogram = os.path.join(abs_path, output_dir_spectrogram)
output_dir_res_spectrogram = os.path.join(abs_path, output_dir_res_spectrogram)

# or list dir
residual_files = os.listdir(output_dir_res_audio)


def all_plot(mix, noise, true_perc, output_wave, residual):
    plt.figure(figsize=(15, 10))
    plt.subplot(5, 2, 1)
    plt.plot(noise[0])
    plt.title(f'Noise file : {noise_names[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    print(f'File path: {os.path.join(output_dir_audio, noise_names[i])}')

    plt.subplot(5, 2, 2)
    stft = get_stft(noise)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Noise with Class : {}'.format(classes[i]))

    plt.subplot(5, 2, 3)
    plt.plot(true_perc[0])
    plt.title('True percussion File: ' +
              f'{percussion_files[i]} with value k = {k_values[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    print(f'File path: {os.path.join(output_dir_audio, percussion_files[i])}')

    plt.subplot(5, 2, 4)
    stft = get_stft(true_perc)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('True percussion')

    plt.subplot(5, 2, 5)
    plt.plot(mix[0])
    plt.title(f'{mix_names[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    print(f'File path: {os.path.join(output_dir_audio, mix_names[i])}')

    plt.subplot(5, 2, 6)
    stft = get_stft(mix)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mixture')

    plt.subplot(5, 2, 7)
    plt.plot(output_wave[0])
    plt.title('Separated percussion File')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    print(f'File path: Separated percussion {
          os.path.join(output_dir_audio, percussion_files[i])}')

    plt.subplot(5, 2, 8)
    stft = get_stft(output_wave)
    mag = torch.abs(stft)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(
    ), ref=np.max), y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Separated percussion')

    plt.subplot(5, 2, 9)
    plt.plot(residual[0])
    plt.title(f'Residual file')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    print(f'File path: {os.path.join(
        output_dir_res_audio, residual_files[i])}')

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
