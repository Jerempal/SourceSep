# %%
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torchsummary as summary
import librosa.display
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os
import pandas as pd
import torch
from torch.optim import AdamW
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from data.config import *
from data.dataset import MixtureDataset, AudioMixtureDataset
from tqdm import tqdm
from torchlibrosa.stft import STFT, ISTFT, magphase
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from data.utils import save_checkpoint, load_checkpoint

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


# %%


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

    def forward(self, inputs):

        inputs = inputs.unsqueeze(1)

        """ Encoder 1 """
        encoder1 = self.encoder_block1(inputs)
        s = self.shortcut(inputs)
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

        return output, skip3

# %%


# model = ResUNet(in_c=1, out_c=32).to("cuda")
# summary.summary(model, (129, 489), batch_size=16)

# %%


class MultiTaskResUNet(nn.Module):
    def __init__(self, num_noise_classes):
        super().__init__()
        self.resunet = ResUNet(in_c=1, out_c=32)

        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Add max pooling here
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Add max pooling here
            nn.Dropout(0.3),
        )

        # output classifier
        self.classifier_output = nn.Sequential(
            nn.Linear(32 * 8 * 30, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_noise_classes),  # Corrected the input size to 64
        )

    def forward(self, x):
        output, skip3 = self.resunet(x)
        x = self.classifier(skip3)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier_output(x)
        return output, x


# %%
# model = MultiTaskResUNet(num_noise_classes=8).to("cuda")
# summary.summary(model, (129, 489), batch_size=16)


# %%

# Define the multi-task loss function
def multi_task_loss(separation_output, classification_output, true_percussion, true_class, alpha=0.7, beta=0.3):
    mse_loss = nn.MSELoss()

    # Ensure true_class is of type torch.LongTensor
    true_class = true_class.long()

    separation_loss = mse_loss(separation_output, true_percussion)
    classification_loss = nn.CrossEntropyLoss()(classification_output, true_class)

    loss = alpha * separation_loss + beta * classification_loss
    
    print(f"Separation Loss: {separation_loss.item()}, Classification Loss: {classification_loss.item()}")
    
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
dataset = AudioMixtureDataset(metadata_file=metadata, k=None, noise_class=None)


# Create train, validation, and test splits
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(
    indices, test_size=0.2, random_state=42)
train_indices, test_indices = train_test_split(
    train_indices, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Save these indices
np.save('train_indices.npy', train_indices)
np.save('val_indices.npy', val_indices)
np.save('test_indices.npy', test_indices)

# Create data loaders
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=32)
val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=32)
test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=32)

# %%

train_indices = np.load('train_indices.npy')
val_indices = np.load('val_indices.npy')
test_indices = np.load('test_indices.npy')

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=32)
val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=32)
test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=32)


# %%
# test the data loaders
for batch in train_loader:
    print(batch['mixture_audio'].shape)
    print(batch['percussion_audio'].shape)
    print(batch['noise_class'].shape)
    break

print(batch['noise_class'])

print(len(train_loader))  # 97
print(len(val_loader))  # 13
print(len(test_loader))  # 13

# %%


class SpectrogramReconstructor:
    def __init__(self):
        pass

    def magphase(self, real, imag):
        mag = (real ** 2 + imag ** 2) ** 0.5
        cos = real / torch.clamp(mag, 1e-10, np.inf)
        sin = imag / torch.clamp(mag, 1e-10, np.inf)
        return mag, cos, sin

    def reconstruct(self, mag_mask, real_mask, imag_mask):

        _, mask_cos, mask_sin = self.magphase(real_mask, imag_mask)

        # calculate the |Y| = |M| * |X|
        estimated_mag = mag_mask * mix_mag

        # Reconstruct the complex spectrogram
        Y_real = estimated_mag * (mask_cos * mix_cos - mask_sin * mix_sin)
        Y_imag = estimated_mag * (mask_cos * mix_sin + mask_sin * mix_cos)
        Y_complex = torch.complex(Y_real, Y_imag)

        return Y_complex


# ISTFT conversion function


def istft(y_complex, n_fft, hop_length):

    y = torch.istft(
        y_complex, n_fft, hop_length, window=torch.hann_window(256, device='cuda'), length=31248)

    return y


# %%

# Define the model, optimizer and loss function
model = MultiTaskResUNet(num_noise_classes=8).to('cuda')
optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True)
criterion = multi_task_loss
device = "cuda"


# %%
# Train the model

train_losses = []
val_losses = []
best_val_loss = np.inf
patience = 5
num_epochs = 10

# model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer)

start_epoch = 0

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {
                     epoch + 1}/{num_epochs} Training Loss: {train_loss:.4f}", colour='green')
    for i, batch in enumerate(train_bar):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Move data to device
        mixture = batch['mixture_audio'].to(device)
        true_percussion = batch['percussion_audio'].to(device)
        true_class = batch['noise_class'].to(device)

        # Calculate real and imaginary parts of the mixture
        mix_stft = torch.stft(mixture, n_fft=256, hop_length=64, win_length=256, window=torch.hann_window(
            window_length=256, device=device), return_complex=True)
        mix_mag, mix_cos, mix_sin = magphase(mix_stft.real, mix_stft.imag)

        # Forward pass
        output, class_output = model(mix_mag)

        mag_mask = torch.sigmoid(output[:, 0, :, :])
        real_mask = torch.tanh(output[:, 1, :, :])
        imag_mask = torch.tanh(output[:, 2, :, :])

        # Reconstruct the complex spectrogram
        Y_complex = SpectrogramReconstructor().reconstruct(mag_mask, real_mask, imag_mask)
        percussion_sep = torch.istft(Y_complex, n_fft=256, hop_length=64, win_length=256,
                                     window=torch.hann_window(256, device=device), length=31248)

        # Calculate the accuracy
        _, predicted = torch.max(class_output, 1)
        total += true_class.size(0)
        correct += (predicted == true_class).sum().item()

        # Calculate the loss
        loss = criterion(percussion_sep, class_output,
                         true_percussion, true_class)

        # Backward pass
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        train_bar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} Training Loss: {train_loss/(i+1):.4f}")

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    accuracy = correct / total

    # Validation
    model.eval()
    val_loss = 0
    total = 0
    correct = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {
                   epoch + 1}/{num_epochs} Validation Loss: {val_loss:.4f}", colour='red')
    with torch.no_grad():
        for i, batch in enumerate(val_bar):
            # Move data to device
            mixture = batch['mixture_audio'].to(device)
            true_percussion = batch['percussion_audio'].to(device)
            true_class = batch['noise_class'].to(device)

            # Calculate real and imaginary parts of the mixture
            mix_stft = torch.stft(mixture, n_fft=256, hop_length=64, win_length=256, window=torch.hann_window(
                window_length=256, device=device), return_complex=True)
            mix_mag, mix_cos, mix_sin = magphase(mix_stft.real, mix_stft.imag)

            # Forward pass
            output, class_output = model(mix_mag)

            mag_mask = torch.sigmoid(output[:, 0, :, :])
            real_mask = torch.tanh(output[:, 1, :, :])
            imag_mask = torch.tanh(output[:, 2, :, :])

            # Reconstruct the complex spectrogram
            Y_complex = SpectrogramReconstructor().reconstruct(mag_mask, real_mask, imag_mask)
            percussion_sep = torch.istft(Y_complex, n_fft=256, hop_length=64, win_length=256,
                                         window=torch.hann_window(256, device=device), length=31248)

            # Calculate the accuracy
            _, predicted = torch.max(class_output, 1)
            total += true_class.size(0)
            correct += (predicted == true_class).sum().item()

            # Calculate the loss
            loss = criterion(percussion_sep, class_output,
                             true_percussion, true_class)

            val_loss += loss.item()
            val_bar.set_description(
                f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss/(i+1):.4f}")

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = correct / total

    print(f"Epoch {epoch + 1}/{num_epochs} Training Loss: {train_loss:.4f}, Training Accuracy: {
          accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save checkpoint at the end of each epoch or based on some condition
    save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir='checkpoint',
                    filename='checkpoint_epoch_{}.pth'.format(epoch))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 5
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model improved. Saving the model")
    else:
        patience -= 1
        if patience == 0:
            print("Early stopping")
            break

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
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model.eval()
test_loss = 0
test_loss_list = []
all_preds = []
all_labels = []

test_bar = tqdm(test_loader, desc="Testing", colour='yellow')
with torch.no_grad():
    for i, batch in enumerate(test_bar):
        # Move data to device
        mixture = batch['mixture_audio'].to(device)
        true_percussion = batch['percussion_audio'].to(device)
        true_class = batch['noise_class'].to(device)

        # Calculate real and imaginary parts of the mixture
        mix_stft = torch.stft(mixture, n_fft=256, hop_length=64, win_length=256, window=torch.hann_window(
            window_length=256, device=device), return_complex=True)
        mix_mag, mix_cos, mix_sin = magphase(mix_stft.real, mix_stft.imag)

        # Forward pass
        output, class_output = model(mix_mag)

        mag_mask = torch.sigmoid(output[:, 0, :, :])
        real_mask = torch.tanh(output[:, 1, :, :])
        imag_mask = torch.tanh(output[:, 2, :, :])

        # Reconstruct the complex spectrogram
        Y_complex = SpectrogramReconstructor().reconstruct(mag_mask, real_mask, imag_mask)
        percussion_sep = torch.istft(Y_complex, n_fft=256, hop_length=64, win_length=256,
                                     window=torch.hann_window(256, device=device), length=31248)

        # Calculate the accuracy
        _, predicted = torch.max(class_output, 1)
        total += true_class.size(0)
        correct += (predicted == true_class).sum().item()

        # Calculate the loss
        loss = criterion(percussion_sep, class_output,
                         true_percussion, true_class)

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

labels = ['air_conditioner', 'car_horn', 'children_playing',
          'dog_bark', 'drilling', 'engine_idling', 'siren', 'jackhammer']
cm = confusion_matrix(all_labels, all_preds, labels=range(8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation='vertical')
plt.show()

# classification report
print(classification_report(all_labels, all_preds, target_names=labels))

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
    librosa.display.specshow(mag_mask[i].cpu().detach().numpy(), sr=7812, hop_length=64, x_axis='time', y_axis='linear', ax=ax[1])
    ax[1].set_title('Magnitude Mask')
    librosa.display.specshow(real_mask[i].cpu().detach().numpy(), sr=7812, hop_length=64, x_axis='time', y_axis='linear', ax=ax[2])
    ax[2].set_title('Real Mask')
    
plt.tight_layout()
plt.show()

# %%

# see the predictions of the last batch by plotting the audio wtih the predicted class and the true class

plt.figure(figsize=(20, 20))
for i in range(mixture.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.plot(mixture[i])
    plt.title(f'Mixture with {labels[true_class[i]]} and predicted class {labels[all_preds[i]]}')
plt.tight_layout()
plt.show()

    
# %%
