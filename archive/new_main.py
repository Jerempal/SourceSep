# %%
import torchsummary as summary
import librosa.display
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from data.config import *
from data.utils import *
# from data.dataset import MixtureDataset, AudioMixtureDataset
from data.dataset import AudioDataset
from tqdm import tqdm
from torchlibrosa.stft import STFT, ISTFT, magphase
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from metrics_loss import *

# load neccessary metrics/confusion matrix for multilabel classification
from sklearn.metrics import multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# mp.set_start_method('spawn', force=True)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(ResidualBlock, self).__init__()

        self.residual_block = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_c, out_c,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c,
                      kernel_size=3, stride=1, padding=1),
        )

        """ Shortcut Connection """
        self.shortcut = nn.Conv2d(
            in_c, out_c, kernel_size=1, stride=stride, padding=0)

    def forward(self, inputs):
        x = self.residual_block(inputs)
        s = self.shortcut(inputs)

        skip = x + s
        return skip


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderBlock, self).__init__()
        self.upsampling = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=2, stride=2, padding=0, dilation=1)
        self.residual_block = ResidualBlock(
            out_c * 2, out_c)
        # self.upsampling = nn.Upsample(
        #     scale_factor=2, mode='bilinear', align_corners=True)
        # self.residual_block = ResidualBlock(
        #     in_c + out_c, out_c)

    def forward(self, x, skip):
        # Upsample
        x = self.upsampling(x)
        # Ensure x and skip have the same spatial dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)

        # Concatenate
        x = torch.cat([x, skip], dim=1)

        # Residual block
        x = self.residual_block(x)

        return x


class ResUNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResUNet, self).__init__()

        """ Encoder 1 """
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_c, out_c,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c,
                      kernel_size=3, stride=1, padding=1),
        )

        """ Shortcut Connection """
        self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

        """ Encoder 2 and 3"""
        self.encoder_block2 = ResidualBlock(
            out_c, out_c * 2, stride=2)
        self.encoder_block3 = ResidualBlock(
            out_c * 2, out_c * 4, stride=2)

        """ Bridge """
        self.bridge = ResidualBlock(
            out_c * 4, out_c * 8, stride=2)

        """ Decoder """
        self.decoder_block1 = DecoderBlock(out_c * 8, out_c * 4)
        self.decoder_block2 = DecoderBlock(out_c * 4, out_c * 2)
        self.decoder_block3 = DecoderBlock(out_c * 2, out_c)

        """ Output """
        self.output = nn.Sequential(
            nn.Conv2d(out_c, 3, kernel_size=1, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            # Linear
            nn.Flatten(),
            nn.Linear(32 * 8 * 30, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 8)
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        """ Encoder 1 """
        encoder1 = self.encoder_block1(x)
        s = self.shortcut(x)
        skip1 = encoder1 + s

        """ Encoder 2 and 3 """
        skip2 = self.encoder_block2(skip1)
        skip3 = self.encoder_block3(skip2)

        """ Bridge """
        bridge = self.bridge(skip3)

        """ Decoder """
        decoder1 = self.decoder_block1(bridge, skip3)
        decoder2 = self.decoder_block2(decoder1, skip2)
        decoder3 = self.decoder_block3(decoder2, skip1)

        """ Output """
        output = self.output(decoder3)

        output_masks_dict = {
            'mag_mask': torch.sigmoid(output[:, 0, :, :]),
            'real_mask': torch.tanh(output[:, 1, :, :]),
            'imag_mask': torch.tanh(output[:, 2, :, :])
        }

        class_output = self.classifier(skip3)
        # return output, class_output
    
        return output_masks_dict, class_output


# # %%
# model = ResUNet(in_c=1, out_c=32).to('cuda')

# summary.summary(model, (129, 489), batch_size=12)

# %%

# class MultiTaskResUNet(nn.Module):
#     def __init__(self, num_noise_classes):
#         super().__init__()
#         self.resunet = ResUNet(in_c=1, out_c=32)

#         # Classification head
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_noise_classes),
#         )

#     def forward(self, x):

#         output, skip3 = self.resunet(x)
#         x = self.classifier(skip3)

#         return output, x

# %%

# Define the multi-task loss function


def multi_task_loss(separation_output, classification_output, true_percussion, true_class, alpha=0.7, beta=0.3, spectrogram_loss=False):

    if spectrogram_loss == False:
        mse_loss = nn.MSELoss()
        separation_loss = mse_loss(separation_output, true_percussion)

    else:
        separation_loss = spectral_loss(separation_output, true_percussion)

    # classification_loss = nn.CrossEntropyLoss()(classification_output, true_class)
    classification_loss = nn.BCEWithLogitsLoss()(classification_output, true_class)

    loss = alpha * separation_loss + beta * classification_loss

    return loss


# %%


# Load metadata
metadata = pd.read_csv(os.path.join(
    DATASET_MIX_AUDIO_PATH, "metadata.csv"))

# define the train, validation and test sets

# dataset = MixtureDataset(metadata_file=metadata, k=0.6,
#                          noise_class=None)
# dataset = AudioMixtureDataset(metadata_file=metadata, k=0.4,
#                               noise_class='siren')
# dataset = AudioMixtureDataset(metadata_file=metadata, k=None, noise_class=None)

# dataset = AudioDataset(metadata_file=metadata, noise_classes=[
#                                           'engine_idling', 'air_conditioner'], random_noise=True)
dataset = AudioDataset(metadata_file=metadata, random_noise=True)

# %%
# Create train, validation, and test splits
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(
    indices, test_size=0.2, random_state=42)
train_indices, test_indices = train_test_split(
    train_indices, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Save these indices
# np.save('train_indices_engine_air.npy', train_indices)
# np.save('val_indices_engine_air.npy', val_indices)
# np.save('test_indices_engine_air.npy', test_indices)

np.save('train_indices.npy', train_indices)
np.save('val_indices.npy', val_indices)
np.save('test_indices.npy', test_indices)

# Create data loaders
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, sampler=train_sampler,
                          batch_size=32, num_workers=8)
val_loader = DataLoader(dataset, sampler=val_sampler,
                        batch_size=32, num_workers=8)
test_loader = DataLoader(dataset, sampler=test_sampler,
                         batch_size=32, num_workers=8)

# %%

# when using the saved indices
train_indices = np.load('train_indices.npy')
val_indices = np.load('val_indices.npy')
test_indices = np.load('test_indices.npy')

# train_indices = np.load('train_indices_engine_air.npy')
# val_indices = np.load('val_indices_engine_air.npy')
# test_indices = np.load('test_indices_engine_air.npy')

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=32, num_workers=2, persistent_workers=True, prefetch_factor=2)
# val_loader = DataLoader(dataset, sampler=val_sampler,
#                         batch_size=32, num_workers=2, persistent_workers=True, prefetch_factor=2)
# test_loader = DataLoader(dataset, sampler=test_sampler,
#                          batch_size=32, num_workers=2, persistent_workers=True, prefetch_factor=2)

train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=40)
val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=40)
test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=12)

# %%


class SpectrogramReconstructor:
    def __init__(self):
        pass

    def magphase(self, real, imag):
        mag = (real ** 2 + imag ** 2) ** 0.5
        cos = real / torch.clamp(mag, 1e-10, np.inf)
        sin = imag / torch.clamp(mag, 1e-10, np.inf)

        return mag, cos, sin

    def reconstruct(self, mag_mask, real_mask, imag_mask, mix_stft):

        mix_mag, mix_cos, mix_sin = self.magphase(mix_stft.real, mix_stft.imag)
        _, mask_cos, mask_sin = self.magphase(real_mask, imag_mask)

        # calculate the |Y| = |M| * |X|
        estimated_mag = mag_mask * mix_mag

        # Reconstruct the complex spectrogram
        Y_real = estimated_mag * (mask_cos * mix_cos - mask_sin * mix_sin)
        Y_imag = estimated_mag * (mask_cos * mix_sin + mask_sin * mix_cos)
        sep_output = torch.complex(Y_real, Y_imag)

        return sep_output


# ISTFT conversion function


def istft(sep_output, n_fft, hop_length):

    y = torch.istft(
        sep_output, n_fft, hop_length, window=torch.hann_window(256, device='cuda'), length=31248)

    return y


# %%

# Define the model, optimizer and loss function
# model = MultiTaskResUNet(num_noise_classes=8).to("cuda")
model = ResUNet(in_c=1, out_c=32).to("cuda")
optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True)
# optimizer = AdamW(model.parameters(), lr=0.001)
criterion = multi_task_loss
device = "cuda"

# %%
# Train the model

train_losses = []
val_losses = []
best_val_loss = np.inf
patience = 5
num_epochs = 5

#%%
# model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, checkpoint_dir='checkpoint', filename='checkpoint_air_engine_epoch_3.pth')
# model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, checkpoint_dir='checkpoint', filename='checkpoint_air_engine_spectralv1_epoch_2.pth')
# model, optimizer, start_epoch, loss = load_checkpoint(
#     model, optimizer, checkpoint_dir='checkpoint', filename='checkpoint_spectral_epoch_2.pth')

start_epoch = 0

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0 
    train_bar = tqdm(train_loader, desc=f"Epoch {
                     epoch + 1}/{num_epochs} Training Loss: {train_loss:.4f}", colour='green')
    for i, batch in enumerate(train_bar):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Move data to device
        # mixture = batch['mixture_audio'].to(device)
        # true_percussion = batch['percussion_audio'].to(device)
        mix_stft = batch['mix_stft'].to(device)
        true_percussion_stft = batch['perc_stft'].to(device)

        # true_class = batch['noise_class'].to(device)
        # ici true class est un tensor de taille (batch_size, 8) avec des 0 et des 1 pour les classes présentes et absentes
        true_class = batch['noise_labels'].to(device)

        # Forward pass
        output, class_output = model(torch.abs(mix_stft))

        # Reconstruct the complex spectrogram
        sep_output = SpectrogramReconstructor().reconstruct(
            output['mag_mask'], output['real_mask'], output['imag_mask'], mix_stft)
        # percussion_sep = istft(sep_output, n_fft=256, hop_length=64)

        # Calculate the loss
        # loss = criterion(percussion_sep, class_output, true_percussion, true_class)
        loss = criterion(sep_output, class_output, true_percussion_stft,
                         true_class, alpha=0.7, beta=0.3, spectrogram_loss=True)

        # Backward pass
        loss.backward()
        optimizer.step()

        # else we calculate log spectral loss so we need to calculate the stft of the separated percussion (sep_output is the complex spectrogram of the separated percussion)
        # true_percussion_stft = torch.stft(true_percussion, n_fft=256, hop_length=64, win_length=256, window=torch.hann_window(window_length=256, device=device), return_complex=True)

        train_loss += loss.item()
        train_bar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} Training Loss: {train_loss/(i+1):.4f}")

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    
    # confusion matrix for multilabel classification
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    accuracy = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {
        epoch + 1}/{num_epochs} Validation Loss: {val_loss:.4f}", colour='red')
    with torch.no_grad():
        for i, batch in enumerate(val_bar):
            # Move data to device
            # mixture = batch['mixture_audio'].to(device)
            mix_stft = batch['mix_stft'].to(device)
            # true_percussion = batch['percussion_audio'].to(device)
            true_percussion_stft = batch['perc_stft'].to(device)
            true_class = batch['noise_labels'].to(device)

            # Forward pass
            output, class_output = model(torch.abs(mix_stft))

            # Reconstruct the complex spectrogram
            sep_output = SpectrogramReconstructor().reconstruct(
                output['mag_mask'], output['real_mask'], output['imag_mask'], mix_stft)
            # percussion_sep = istft(sep_output, n_fft=256, hop_length=64)

            # Calculate the loss
            # loss = criterion(percussion_sep, class_output, true_percussion, true_class)
            loss = criterion(sep_output, class_output, true_percussion_stft,
                             true_class, alpha=0.7, beta=0.3, spectrogram_loss=True)

            val_loss += loss.item()
            
            # Calculate multi-label classification accuracy
            predicted = torch.sigmoid(class_output) > 0.5
            # correct += (predicted == true_class).sum().item
            total += true_class.size(0)
            correct += (predicted == true_class).all(dim=1).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_class.cpu().numpy())
            
            val_bar.set_description(
                f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss/(i+1):.4f}")
            
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        accuracy = correct / total
        
        print(f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        
        # confusion matrix
        labels = ['air_conditioner', 'car_horn', 'children_playing',
                  'dog_bark', 'drilling', 'engine_idling', 'siren', 'jackhammer']
        cm = multilabel_confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[0], display_labels=labels)
        disp.plot(xticks_rotation='vertical')
        plt.show()
        
        # classification report multilabel
        print(classification_report(all_labels, all_preds, target_names=labels))
        
        # save checkpoint
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir='checkpoint',
                        filename='checkpoint_spectralv1_epoch_{}'.format(epoch))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), 'best_model.pth')
            torch.save(model.state_dict(), 'best_model_spectralv1.pth')
            print("Model improved. Saving the model...")

        else:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                break

#%%
# save losses
np.save('train_losses.npy', train_losses)
np.save('val_losses.npy', val_losses)

#%%
#load losses
train_losses = np.load('train_losses.npy')
val_losses = np.load('val_losses.npy')

# %%

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# Test the model
# model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model.load_state_dict(torch.load('best_model_air_engine_spectral.pth'))
model.eval()
test_loss = 0
test_loss_list = []
all_preds = []
all_labels = []
correct = 0
total = 0
test_bar = tqdm(test_loader, desc="Testing", colour='yellow')
with torch.no_grad():
    for i, batch in enumerate(test_bar):
        # Move data to device
        mixture = batch['mixture_audio'].to(device)
        true_percussion = batch['percussion_audio'].to(device)
        mix_stft = batch['mix_stft'].to(device)
        true_percussion_stft = batch['perc_stft'].to(device)
        # true_class = batch['noise_class'].to(device)
        true_class = batch['noise_labels'].to(device)

        # Forward pass
        output, class_output = model(torch.abs(mix_stft))

        mag_mask = output['mag_mask']
        real_mask = output['real_mask']
        imag_mask = output['imag_mask']

        # Reconstruct the complex spectrogram
        sep_output = SpectrogramReconstructor().reconstruct(
            mag_mask, real_mask, imag_mask, mix_stft)
        percussion_sep = torch.istft(sep_output, n_fft=256, hop_length=64, win_length=256,
                                     window=torch.hann_window(256, device=device), length=31248)

        # Calculate the classification accuracy

        # Calculate the loss
        # loss = criterion(percussion_sep, class_output, true_percussion, true_class)
        loss = criterion(sep_output, class_output, true_percussion_stft,
                         true_class, alpha=0.7, beta=0.3, spectrogram_loss=True)

        test_loss += loss.item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(true_class.cpu().numpy())

        test_bar.set_description(
            f"Testing Loss: {test_loss/(i+1):.4f}")

    test_loss /= len(test_loader)
    test_loss_list.append(test_loss)
    accuracy = correct / total
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {accuracy:.4f}")

# %%

# labels = ['air_conditioner', 'car_horn', 'children_playing',
#           'dog_bark', 'drilling', 'engine_idling', 'siren', 'jackhammer']
# cm = confusion_matrix(all_labels, all_preds, labels=range(8))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# disp.plot(xticks_rotation='vertical')
# plt.show()

# # classification report
# print(classification_report(all_labels, all_preds, target_names=labels))

# %%

# listen to the mixture, percussion and the separated percussion

# (batchsize, 31248) 9, 31248
mixture = mixture.cpu().numpy()
percussion = true_percussion.cpu().numpy()
separated_percussion = percussion_sep.cpu().detach().numpy()

# %%
# save them all
for i in range(mixture.shape[0]):
    sf.write(f'mixture_{i}_with_{labels[true_class[i]]}.wav', mixture[i], 7812)
    sf.write(f'percussion_{i}.wav', percussion[i], 7812)
    sf.write(f'separated_percussion_{i}.wav', separated_percussion[i], 7812)


# %%
# plot the spectrogram of the mixture, percussion and separated percussion

for i in range(mixture.shape[0]):
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    # mixture
    mix_stft = librosa.stft(
        mixture[i], n_fft=256, hop_length=64, win_length=256)
    mix_mag, _ = librosa.magphase(mix_stft)
    librosa.display.specshow(librosa.amplitude_to_db(
        mix_mag, ref=np.max), sr=7812, hop_length=64, x_axis='time', y_axis='linear', ax=ax[0])
    ax[0].set_title('Mixture with {}'.format(labels[true_class[i]]))
    # percussion
    perc_stft = librosa.stft(
        percussion[i], n_fft=256, hop_length=64, win_length=256)
    perc_mag, _ = librosa.magphase(perc_stft)
    librosa.display.specshow(librosa.amplitude_to_db(
        perc_mag, ref=np.max), sr=7812, hop_length=64, x_axis='time', y_axis='linear', ax=ax[1])
    ax[1].set_title('Percussion')
    # separated percussion
    sep_stft = librosa.stft(
        separated_percussion[i], n_fft=256, hop_length=64, win_length=256)
    sep_mag, _ = librosa.magphase(sep_stft)
    librosa.display.specshow(librosa.amplitude_to_db(
        sep_mag, ref=np.max), sr=7812, hop_length=64, x_axis='time', y_axis='linear', ax=ax[2])
    ax[2].set_title('Separated Percussion')
    plt.tight_layout()
    plt.show()

# %%
# plot the masks
for i in range(mixture.shape[0]):
    fig, ax = plt.subplots(3, 1, figsize=(20, 20))
    # mixture
    mix_stft = librosa.stft(
        mixture[i], n_fft=256, hop_length=64, win_length=256)
    mix_mag, _ = librosa.magphase(mix_stft)
    librosa.display.specshow(librosa.amplitude_to_db(
        mix_mag, ref=np.max), sr=7812, hop_length=64, x_axis='time', y_axis='linear', ax=ax[0])
    ax[0].set_title('Mixture with {}'.format(labels[true_class[i]]))
    # masks
    librosa.display.specshow(mag_mask[i].cpu().detach().numpy(
    ), sr=7812, hop_length=64, x_axis='time', y_axis='linear', ax=ax[1])
    ax[1].set_title('Magnitude Mask')
    librosa.display.specshow(real_mask[i].cpu().detach().numpy(
    ), sr=7812, hop_length=64, x_axis='time', y_axis='linear', ax=ax[2])
    ax[2].set_title('Real Mask')

plt.tight_layout()
plt.show()

# %%

# see the predictions of the last batch by plotting the audio wtih the predicted class and the true class

plt.figure(figsize=(20, 20))
for i in range(mixture.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.plot(mixture[i])
    plt.title(f'Mixture with {labels[true_class[i]]} and predicted class {
              labels[all_preds[i]]}')
plt.tight_layout()
plt.show()


# %%
