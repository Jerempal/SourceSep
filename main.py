# %%

# %%

import random
import pandas as pd
from data.config import *
from data.dataset import *
from train import train
from data.utils import *
from torch.utils.data import DataLoader
from torch import optim
from model.Last_model import *
from metrics_loss import *
from test import test, process_metadata, DATASET_PREDICTED_AUDIO_PATH, noise_name

# We have 7358 sounds of differents classes (dog bark, drilling, jackhammer, siren, children_playing, engine idling, air conditioner, car horn) with a duration of maximum 4 seconds some are shorter
# 387 files of the percussions class that we want to separate from the others or "hear" better

# Load metadata
metadata = pd.read_csv(os.path.join(
    DATASET_MIX_AUDIO_PATH, "metadata.csv"))

# define the train, validation and test sets

dataset = MixtureDataset(metadata_file=metadata, k=0.8,
                         noise_class='engine_idling')
# Split the dataset into training, validation and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size])

#%%

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# %%
# Define the model, optimizer and loss function
model = ResUNet(in_c=3, out_c=32).to("cuda")
# optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-6, amsgrad=True)
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9)
criterion = loss_wav
# Train the model
train_losses, val_losses, SDRi_list, SISDR_list = train(model, train_loader, val_loader,
                                                        num_epochs=5, optimizer=optimizer, criterion=criterion, device="cuda")

# Plot the training and validation losses
plot_loss(train_losses, val_losses)

# %%

# get the best model
model = ResUNet(in_c=3, out_c=32)
# model.load_state_dict(torch.load(
# 'best_model.4_with_val_loss1.4713155581400945.pth', weights_only=True))
model.load_state_dict(torch.load(
    'best_model.4_with_val_loss0.015820460413031433.pth', weights_only=True))

# test the model
test_loss, SDRi_list, SISDR_list, mixtures_names = test(
    model, test_loader, criterion=loss_wav, device="cuda")

# Process the metadata
df, noise_name = process_metadata(
    metadata, DATASET_PREDICTED_AUDIO_PATH, mixtures_names, noise_name=noise_name)

# %%

# for file in os.listdir(DATASET_PREDICTED_AUDIO_PATH):
#     if file.endswith(".csv"):
#         df = pd.read_csv(os.path.join(DATASET_PREDICTED_AUDIO_PATH, file))
#         break

# # Create the dataset
# test_separation_dataset = TestSeparationDataset(
#     metadata_file=df, k=0.8)

# # Create the data loader
# test_separation_loader = DataLoader(
#     test_separation_dataset, batch_size=16, shuffle=True)

# # Plot the predicted and true percussion signals along their spectrograms
# data = next(iter(test_separation_loader))

# # randomly select an index
# i = random.randint(0, len(data['predicted audio']) - 1)

# # convert audio to numpy
# predicted_audio = data['predicted audio'][i].cpu().detach().numpy()
# true_percussion_audio = data['percussion audio'][i].cpu().detach().numpy()
# noise_audio = data['noise audio'][i].cpu().detach().numpy()
# mixture_audio = data['mixture audio'][i].cpu().detach().numpy()

# # Plot the predicted and true percussion signals
# plt.figure(figsize=(20, 10))
# titles = ["Predicted Percussion Signal", "True Percussion Signal", "Noise Signal with noise level: {:.2f}".format(
#     data['noise level'][i]), "Mixture Signal with noise level: {:.2f}".format(data['noise level'][i])]
# audios = [predicted_audio, true_percussion_audio, noise_audio, mixture_audio]

# for j in range(4):
#     plt.subplot(2, 4, j+1)
#     plt.title(titles[j])
#     librosa.display.waveshow(audios[j], sr=7812)
#     plt.ylim([-1, 1])

# # Plot the predicted and true percussion spectrograms
# for j in range(4):
#     plt.subplot(2, 4, j+5)
#     plt.title(titles[j] + " Spectrogram")
#     spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(
#         audios[j], n_fft=n_fft, hop_length=hop_length)), ref=np.max)
#     librosa.display.specshow(spectrogram, sr=7812, hop_length=hop_length)
#     plt.colorbar(format='%+2.0f dB')

# plt.tight_layout()
# plt.show()

# # show the path of all the files
# print(f'Predicted audio path: {data["path pred"][i]}')
# print(f'True percussion audio path: {data["path true"][i]}')
# print(f'Noise audio path: {data["path noise"][i]}')

# # Save the mixture audio
# sf.write("mixture_audio.wav", mixture_audio, 7812)
# print(f'path of mix saved', os.path.abspath("mixture_audio.wav"))

# # %%
#%%
import soundfile as sf

for file in os.listdir(DATASET_PREDICTED_AUDIO_PATH):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATASET_PREDICTED_AUDIO_PATH, file))
        break

# Create the dataset
test_separation_dataset = TestSeparationDataset(
    metadata_file=df, k=0.8, noise_class=noise_name)

# Create the data loader
test_separation_loader = DataLoader(
    test_separation_dataset, batch_size=16, shuffle=True)

# Plot the predicted and true percussion signals along their spectrograms
data = next(iter(test_separation_loader))

# randomly select an index
i = random.randint(0, len(data['predicted audio']) - 1)

# convert audio to numpy
predicted_audio = data['predicted audio'][i].cpu().detach().numpy()
true_percussion_audio = data['percussion audio'][i].cpu().detach().numpy()
noise_audio = data['noise audio'][i].cpu().detach().numpy()
mixture_audio = data['mixture audio'][i].cpu().detach().numpy()

# Plot the predicted and true percussion signals
plt.figure(figsize=(20, 10))
titles = ["Predicted Percussion Signal", "True Percussion Signal", "Noise Signal with noise level: {:.2f}".format(
    data['noise level'][i]), "Mixture Signal with noise level: {:.2f}".format(data['noise level'][i])]
audios = [predicted_audio, true_percussion_audio, noise_audio, mixture_audio]

for j in range(4):
    plt.subplot(2, 4, j+1)
    plt.title(titles[j])
    librosa.display.waveshow(audios[j], sr=7812)
    plt.ylim([-1, 1])

# Plot the predicted and true percussion spectrograms
for j in range(4):
    plt.subplot(2, 4, j+5)
    plt.title(titles[j] + " Spectrogram")
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(
        audios[j], n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    librosa.display.specshow(spectrogram, sr=7812, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()

# show the path of all the files
print(f'Predicted audio path: {data["path pred"][i]}')
print(f'True percussion audio path: {data["path true"][i]}')
print(f'Noise audio path: {data["path noise"][i]}')

# Save the mixture audio
sf.write("mixture_audio.wav", mixture_audio, 7812)
print(f'path of mix saved', os.path.abspath("mixture_audio.wav"))

# %%
