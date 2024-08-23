# %%
import torchsummary as summary
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
from model.base import Base
# load neccessary metrics/confusion matrix for multilabel classification
from sklearn.metrics import multilabel_confusion_matrix, classification_report

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

        # self.classifier = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(out_c*4, out_c*2, kernel_size=3, padding=1),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.ReLU(),

        #     # Linear
        #     nn.Flatten(),
        #     nn.Linear(out_c*2 * 8 * 30, out_c*2),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(out_c*2, out_c),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(out_c, 8)
        # )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c*4, out_c*2, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Flatten(),
            # nn.Linear(out_c*4 * 16 * 61, 128),
            # nn.Linear(out_c*4 * 8 * 30, 128),
            nn.Linear(out_c*2 * 4 * 15, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8)
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

class ResUNetv2(nn.Module, Base):
    def __init__(self, in_c, out_c):
        super(ResUNetv2, self).__init__()

        window_size = 256
        hop_size = 64
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.output_channels = 1
        self.target_sources_num = 1
        self.K = 3

        # downsample ratio
        self.time_downsample_ratio = 2**3  # number of encoder layers

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

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
        # self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, padding="same"),

        """ Encoder 2 and 3"""
        self.encoder_block2 = ResidualBlock(out_c, out_c * 2, stride=2)
        self.encoder_block3 = ResidualBlock(out_c * 2, out_c * 4, stride=2)

        """ Bridge """
        self.bridge = ResidualBlock(
            out_c * 4, out_c * 8, stride=2)

        """ Decoder """
        self.decoder_block1 = DecoderBlock(out_c * 8, out_c * 4)
        self.decoder_block2 = DecoderBlock(out_c * 4, out_c * 2)
        self.decoder_block3 = DecoderBlock(out_c * 2, out_c)

        """ Output """
        # self.last_layer = nn.Sequential(
        #     # nn.Conv2d(out_c, 1, kernel_size=1, padding='same'),
        #     nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
        # )

        self.after_conv = nn.Conv2d(
            in_channels=out_c,
            out_channels=self.output_channels * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])

        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)

        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos -
            sin_in[:, None, :, :, :] * mask_sin
        )
        print(out_cos.shape)
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos +
            cos_in[:, None, :, :, :] * mask_sin
        )

        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)

        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (batch_size * self.target_sources_num *
                 self.output_channels, 1, time_steps, freq_bins)

        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)
        return waveform

    def forward(self, mixtures):
        """
        Args:
            input: (batch_size, segment_samples)

        Outputs:
            output_dict: {
            'wav': (batch_size, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag
        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)
                ) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))

        """(batch_size, channels, padded_time_steps, freq_bins)"""
        # Let frequency bins be evenly divided by 2, e.g., 489 -> 488.
        x = x[..., 0: x.shape[-1] - 1]  # (bs, channels, T, F)
        # UNet
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
        # output = self.last_layer(decoder3)

        x = self.after_conv(decoder3)

        # (batch_size, target_sources_num * output_channels * self.K, T, F')

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]
        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)

        output_dict = {'waveform': separated_audio}

        return output_dict


# # %%

# model = ResUNetv2(in_c=1, out_c=16).to("cuda")
# summary.summary(model, (1, 31248), batch_size=20)


# %%
# Define the multi-task loss function


def multi_task_loss(separation_output, classification_output, true_percussion, true_class, alpha=0.7, beta=0.3, spectrogram_loss=False):

    if spectrogram_loss == False:
        mse_loss = nn.MSELoss()
        separation_loss = mse_loss(separation_output, true_percussion)

    else:
        separation_loss = spectral_loss(separation_output, true_percussion)

    # classification_loss = nn.CrossEntropyLoss()(classification_output, true_class)
    classification_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1, 2, 1, 1, 1, 1, 1, 1]).to("cuda"))(
        classification_output, true_class)

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
dataset = AudioDataset(metadata_file=metadata, random_noise=True, classify=False)

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

train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=20)
val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=20)
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
# model = ResUNet(in_c=1, out_c=16).to("cuda")
model = ResUNetv2(in_c=1, out_c=16).to("cuda")
optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True, fused=True)
# optimizer = AdamW(model.parameters(), lr=0.001)
# criterion = multi_task_loss
criterion = nn.MSELoss()
device = "cuda"

# %%
# Train the model

train_losses = []
val_losses = []
best_val_loss = np.inf
patience = 5
num_epochs = 5

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
        mixture = batch['mixture_audio'].to(device)
        true_percussion = batch['percussion_audio'].to(device)
        # mix_stft = batch['mix_stft'].to(device)
        # true_percussion_stft = batch['perc_stft'].to(device)

        # true_class = batch['noise_class'].to(device)
        # ici true class est un tensor de taille (batch_size, 8) avec des 0 et des 1 pour les classes présentes et absentes
        # true_class = batch['noise_labels'].to(device)

        # Forward pass
        # output, class_output = model(torch.abs(mix_stft))
        output_waveform = model(mixture)
        output_waveform = output_waveform['waveform']
        
        # Reconstruct the complex spectrogram
        # sep_output = SpectrogramReconstructor().reconstruct(
        #     output['mag_mask'], output['real_mask'], output['imag_mask'], mix_stft)
        # percussion_sep = istft(sep_output, n_fft=256, hop_length=64)

        # Calculate the loss
        # loss = criterion(percussion_sep, class_output, true_percussion, true_class)
        # loss = criterion(sep_output, class_output, true_percussion_stft,
        #                  true_class, alpha=0.7, beta=0.3, spectrogram_loss=True)

        loss = criterion(output_waveform, true_percussion)
        
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
    # correct = 0
    # total = 0
    # all_preds = []
    # all_labels = []

    val_bar = tqdm(val_loader, desc=f"Epoch {
        epoch + 1}/{num_epochs} Validation Loss: {val_loss:.4f}", colour='red')
    with torch.no_grad():
        for i, batch in enumerate(val_bar):
            # Move data to device
            mixture = batch['mixture_audio'].to(device)
            # mix_stft = batch['mix_stft'].to(device)
            true_percussion = batch['percussion_audio'].to(device)
            # true_percussion_stft = batch['perc_stft'].to(device)
            # true_class = batch['noise_labels'].to(device)

            # Forward pass
            # output, class_output = model(torch.abs(mix_stft))
            output_waveform = model(mixture)
        
            # Reconstruct the complex spectrogram
            # sep_output = SpectrogramReconstructor().reconstruct(
            #     output['mag_mask'], output['real_mask'], output['imag_mask'], mix_stft)
            # percussion_sep = istft(sep_output, n_fft=256, hop_length=64)

            # Calculate the loss
            # loss = criterion(percussion_sep, class_output, true_percussion, true_class)
            # loss = criterion(sep_output, class_output, true_percussion_stft,
            #                  true_class, alpha=0.7, beta=0.3, spectrogram_loss=True)

            loss = criterion(output_waveform, true_percussion)
            val_loss += loss.item()

            # # Calculate multi-label classification accuracy
            # predicted = (torch.sigmoid(class_output) > 0.5).float()
            # # Total for a multi-label classification:
            # total += true_class.size(0) * true_class.size(1)
            # correct += (predicted == true_class).float().sum().item()

            # all_preds.extend(predicted.cpu().numpy())
            # all_labels.extend(true_class.cpu().numpy())

            val_bar.set_description(
                f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss/(i+1):.4f}")

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        # accuracy = correct / total

        # print(f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {
        #       val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        # # confusion matrix
        # labels = ['air_conditioner', 'car_horn', 'children_playing',
        #           'dog_bark', 'drilling', 'engine_idling', 'siren', 'jackhammer']
        # cm = multilabel_confusion_matrix(all_labels, all_preds)

        # # plot confusion matrix
        # from sklearn.metrics import ConfusionMatrixDisplay
        # fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        # for i in range(8):
        #     disp = ConfusionMatrixDisplay(cm[i], display_labels=[0, 1])
        #     disp.plot(ax=ax[i//4, i % 4])
        #     disp.ax_.set_title(labels[i])
        # plt.show()

        # # classification report multilabel
        # print(classification_report(all_labels, all_preds,
        #       target_names=labels, zero_division=0))

        # save checkpoint
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir='checkpoint',
                        filename='checkpoint_v4_epoch_{}'.format(epoch))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), 'best_model.pth')
            torch.save(model.state_dict(), 'best_model_v4.pth')
            print("Model improved. Saving the model...")

        else:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                break


# %%
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# %%
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss

# on utilise precision_recall_fscore_support pour calculer les scores de précision, recall et f1-score (multi-label)
# accuracy_score pour calculer l'accuracy (multi-label) 
# hamming_loss pour calculer la hamming loss (multi-label)

# on utilise classification_report pour afficher les résultats
# on utilise multilabel_confusion_matrix pour afficher les matrices

# Test the model
model.eval()
test_loss = 0
test_losses = []
# confusion matrix for multilabel classification
correct = 0
total = 0
all_preds = []
all_labels = []

test_bar = tqdm(test_loader, desc=f"Testing Loss: {test_loss:.4f}", colour='red')
with torch.no_grad():
    for i, batch in enumerate(test_bar):
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

        test_loss += loss.item()

        # Calculate multi-label classification accuracy
        predicted = (torch.sigmoid(class_output) > 0.5).float()
        # Total for a multi-label classification:
        total += true_class.size(0) * true_class.size(1)
        correct += (predicted == true_class).float().sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(true_class.cpu().numpy())

        test_bar.set_description(
            f"Testing Loss: {test_loss/(i+1):.4f}")

    test_loss /= len(test_loader)   
    test_losses.append(test_loss)
    accuracy = correct / total
    
#%%
# accuracy score multi-label
accuracy_m = accuracy_score(all_labels, all_preds)
# hamming loss multi-label
hamming = hamming_loss(all_labels, all_preds)
# precision, recall, f1-score multi-label
precision, recall, f1, true_sum = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {accuracy:.4f}")
print(f"Testing Accuracy (multi-label): {accuracy_m:.4f}")
print(f"Hamming Loss (multi-label): {hamming:.4f}")
print(f"Precision (multi-label): {precision:.4f}")
print(f"Recall (multi-label): {recall:.4f}")
print(f"F1-score (multi-label): {f1:.4f}")

# confusion matrix
labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'siren', 'jackhammer']
cm = multilabel_confusion_matrix(all_labels, all_preds)

# plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for i in range(8):
    disp = ConfusionMatrixDisplay(cm[i], display_labels=[0, 1])
    disp.plot(ax=ax[i//4, i % 4])
    disp.ax_.set_title(labels[i])
plt.show()

# classification report multilabel
print(classification_report(all_labels, all_preds, target_names=labels, zero_division=0))

# save the results
results = {
    'test_loss': test_loss,
    'accuracy': accuracy,
    'accuracy_m': accuracy_m,
    'hamming': hamming,
    'precision': precision,
    'recall': recall,
    'f1': f1
}

#%%
np.save('results_spectralv2.npy', results)


# %%

# inference on a single audio file
# load the model
model = ResUNet(in_c=1, out_c=16).to("cuda")
model.load_state_dict(torch.load('best_model_spectralv2.pth'))
model.eval()

# load a test file from the test loader
test_file = next(iter(test_loader))
mix_stft = test_file['mix_stft'].to(device)
mixture_audio = test_file['mixture_audio'].to(device)
percussion_audio = test_file['percussion_audio'].to(device)
true_percussion_stft = test_file['perc_stft'].to(device)
true_class = test_file['noise_labels'].to(device)

# forward pass
output, class_output = model(torch.abs(mix_stft))

# reconstruct the complex spectrogram
sep_output = SpectrogramReconstructor().reconstruct(
    output['mag_mask'], output['real_mask'], output['imag_mask'], mix_stft)
percussion_sep = istft(sep_output, n_fft=256, hop_length=64)

# calculate the loss
loss = criterion(sep_output, class_output, true_percussion_stft,
                 true_class, alpha=0.7, beta=0.3, spectrogram_loss=True)

# calculate multi-label classification accuracy
predicted = (torch.sigmoid(class_output) > 0.5).float()
# total for a multi-label classification:
total = true_class.size(0) * true_class.size(1)
correct = (predicted == true_class).float().sum().item()

# confusion matrix for multilabel classification
all_preds = []
all_labels = []
all_preds.extend(predicted.cpu().numpy())
all_labels.extend(true_class.cpu().numpy())

# accuracy score for correctly predicting positive and negative classes
accuracy_m = accuracy_score(all_labels, all_preds)

# hamming loss multi-label
hamming = hamming_loss(all_labels, all_preds)
# precision, recall, f1-score multi-label
precision, recall, f1, true_sum = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

print(f"Testing Loss: {loss:.4f}, Testing Accuracy: {correct/total:.4f}")
print(f"Testing Accuracy (multi-label): {accuracy_m:.4f}")
print(f"Hamming Loss (multi-label): {hamming:.4f}")
print(f"Precision (multi-label): {precision:.4f}")
print(f"Recall (multi-label): {recall:.4f}")
print(f"F1-score (multi-label): {f1:.4f}")

# confusion matrix
labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'siren', 'jackhammer']
cm = multilabel_confusion_matrix(all_labels, all_preds)

# plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for i in range(8):
    disp = ConfusionMatrixDisplay(cm[i], display_labels=[0, 1])
    disp.plot(ax=ax[i//4, i % 4])
    disp.ax_.set_title(labels[i])
plt.show()

# classification report multilabel
print(classification_report(all_labels, all_preds, target_names=labels, zero_division=0))


# %%

# plot percussions and separated percussions
plt.figure(figsize=(10, 15))
for i in range(3):
    
    plt.subplot(3, 1, 1)
    plt.title('Mixture')
    plt.plot(mixture_audio[i].cpu().numpy())
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 2)
    plt.title('Separated Percussion')
    plt.plot(percussion_sep[i].detach().cpu().numpy())
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 3)
    plt.title('True Percussion')
    plt.plot(percussion_audio[i].cpu().numpy()) 
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    



# %%
k = np.random.randint(0, 12)

# plot the mixture, separated percussion and true percussion
plt.figure(figsize=(10, 15))
plt.subplot(3, 1, 1)
plt.title('Mixture')
plt.plot(mixture_audio[k].cpu().numpy())
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.title('Separated Percussion')
plt.plot(percussion_sep[k].detach().cpu().numpy())
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.title('True Percussion')
plt.plot(percussion_audio[k].cpu().numpy())
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
# %%
k = np.random.randint(0, 12)

# Data and titles for plotting
audio_data = [
    (mixture_audio, 'Mixture'),
    (percussion_audio, 'True Percussion'),
    (percussion_sep.detach(), 'Separated Percussion')
]

spectrogram_data = [
    (mix_stft, 'Mixture Spectrogram'),
    (true_percussion_stft, 'True Percussion Spectrogram'),
    (sep_output, 'Separated Percussion Spectrogram')
]

plt.figure(figsize=(10, 15))

# Plot audio waveforms
for i, (data, title) in enumerate(audio_data):
    plt.subplot(3, 2, 2 * i + 1)
    plt.title(title)
    plt.plot(data[k].cpu().numpy())
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

# Plot spectrograms
for i, (data, title) in enumerate(spectrogram_data):
    plt.subplot(3, 2, 2 * (i + 1))
    plt.title(title)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(data[k].detach().cpu().numpy()), ref=np.max), 
                             y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)

plt.tight_layout()
plt.show()

# %%

def energy_ratio(true_audio, separated_audio):
    true_energy = np.sum(true_audio ** 2)
    separated_energy = np.sum(separated_audio ** 2)
    ratio = separated_energy / true_energy
    return ratio

energy_ratios = []
for i in range(12):
    energy_ratios.append(energy_ratio(percussion_audio[i].cpu().numpy(), percussion_sep[i].detach().cpu().numpy()))
    
energy_ratios = np.array(energy_ratios)
print(f"Energy Ratios: {energy_ratios}")

highest_ratio_indices = np.argsort(energy_ratios)[-3:]
for i in highest_ratio_indices:
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.title('Separated Percussion')
    plt.plot(percussion_sep[i].detach().cpu().numpy())
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 1, 2)
    plt.title('True Percussion')
    plt.plot(percussion_audio[i].cpu().numpy())
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
#%%
lowest_ratio_indices = np.argsort(energy_ratios)[:3]
for i in lowest_ratio_indices:
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.title('Separated Percussion')
    plt.plot(percussion_sep[i].detach().cpu().numpy())
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 1, 2)
    plt.title('True Percussion')
    plt.plot(percussion_audio[i].cpu().numpy())
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()



# %%
#output the magnitude mask and the separated percussion spectrogram
plt.figure(figsize=(10, 5))
plt.title('Magnitude Mask')
plt.imshow(output['mag_mask'][1].detach().cpu().numpy(), aspect='auto', origin='lower')
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 5))
plt.title('Separated Percussion Spectrogram')
librosa.display.specshow(librosa.amplitude_to_db(np.abs(sep_output[1].detach().cpu().numpy()), ref=np.max), 
                         y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 5))
plt.title('Percussion Spectrogram')
librosa.display.specshow(librosa.amplitude_to_db(np.abs(true_percussion_stft[1].detach().cpu().numpy()), ref=np.max), 
                         y_axis='linear', x_axis='time', sr=7812, n_fft=256, hop_length=64)
plt.colorbar()
plt.show()

# %%
