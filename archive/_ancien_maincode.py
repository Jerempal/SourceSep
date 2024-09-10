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
from config import *
from data.utils import save_checkpoint, load_checkpoint
# from data.dataset import MixtureDataset, AudioMixtureDataset
# from data.dataset import AudioDataset
from dataset import PreComputedMixtureDataset
from tqdm import tqdm
from torchlibrosa.stft import STFT, ISTFT, magphase
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from metrics_loss import *
from model.base import Base

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
        # self.target_sources_num = 1
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
            # self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        # mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        # _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        # _mask_imag = torch.tanh(x[:, :, :, 2, :, :])

        mask_mag = torch.sigmoid(x[:, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, 2, :, :])
        # print(mask_mag.shape)
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # print(mask_mag.shape)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, :, :, :] * mask_cos -
            sin_in[:, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, :, :, :] * mask_cos +
            cos_in[:, :, :, :] * mask_sin
        )
        # print(out_cos.shape)
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, :, :, :] * mask_mag)
        # print(out_mag.shape)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # print(out_real.shape)
        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        # shape = (batch_size * self.target_sources_num *
        #  self.output_channels, 1, time_steps, freq_bins)

        shape = (batch_size * self.output_channels, 1, time_steps, freq_bins)
        shape = (batch_size, 1, time_steps, freq_bins)
        # print(shape)

        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)
        # print(out_real.shape)
        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        # waveform = x.reshape(batch_size, self.target_sources_num * self.output_channels, audio_length)
        waveform = x.reshape(batch_size, self.output_channels, audio_length)
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


# %%

# model = ResUNetv2(in_c=1, out_c=16).to("cuda")
# summary.summary(model, (1, 31248), batch_size=20)
# #%%
# input = torch.randn(20, 1, 31248).to("cuda")
# output = model(input)


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
# dataset = AudioDataset(metadata_file=metadata, random_noise=True, classify=False)

# dataset = PreComputedMixtureDataset(metadata_file=os.path.join(
#     DATASET_MIX_AUDIO_PATH, "metadata.csv"))


dataset = PreComputedMixtureDataset(metadata_file=metadata)


# %%
# Create train, validation, and test splits
# indices = list(range(len(dataset)))
# train_indices, val_indices = train_test_split(
#     indices, test_size=0.2, random_state=42)
# train_indices, test_indices = train_test_split(
#     train_indices, test_size=0.30, random_state=42)  # 0.25 * 0.8 = 0.2

# np.save('train_indices_new_last.npy', train_indices)
# np.save('val_indices_new_last.npy', val_indices)
# np.save('test_indices_new_last.npy', test_indices)

# # Create data loaders
# train_sampler = SubsetRandomSampler(train_indices)
# val_sampler = SubsetRandomSampler(val_indices)
# test_sampler = SubsetRandomSampler(test_indices)

# train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=25)
# val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=25)
# test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=25)

# Group the metadata by percussion file
grouped_metadata = metadata.groupby('percussion_file')

# Get all unique percussion files
unique_perc_files = metadata['percussion_file'].unique()

# Split based on percussion files, not individual mixtures
train_perc_files, test_perc_files = train_test_split(unique_perc_files, test_size=0.2, random_state=42)
train_perc_files, val_perc_files = train_test_split(train_perc_files, test_size=0.25, random_state=42)  # 0.25 of the remaining for validation

# Create train, validation, and test datasets by filtering the metadata
train_metadata = metadata[metadata['percussion_file'].isin(train_perc_files)]
val_metadata = metadata[metadata['percussion_file'].isin(val_perc_files)]
test_metadata = metadata[metadata['percussion_file'].isin(test_perc_files)]

# Save the indices (if needed)
train_indices = train_metadata.index.tolist()
val_indices = val_metadata.index.tolist()
test_indices = test_metadata.index.tolist()

np.save('train_indices_new_last.npy', train_indices)
np.save('val_indices_new_last.npy', val_indices)
np.save('test_indices_new_last.npy', test_indices)

# Use these new indices for your DataLoader
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=25)
val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=25)
test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=25)



#%%
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
data = next(iter(train_loader))

for i in range(5):
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.plot(data['mixture_audio'][i,0].numpy())
    plt.title(f'Mixture {i}')
    plt.subplot(3, 1, 2)
    plt.plot(data['percussion_audio'][i,0].numpy())
    plt.title(f'Percussion {i} with k = {data["k"][i]}')
    plt.subplot(3, 1, 3)
    plt.plot(data['noise_audio'][i,0].numpy())
    plt.title(f'Noise {i} with k = {1-data["k"][i]}, noise class = {data["noise_classes"][i]}')
    
    # print path of mix and noise
    print(os.path.join(DATASET_MIX_AUDIO_PATH, data['mix_name'][i]))
    print(os.path.join(DATASET_MIX_AUDIO_PATH, data['noise_file'][i]))
    
    
    plt.tight_layout()
    plt.show()
    
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

train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=32)
val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=32)
test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=32)

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
model = ResUNetv2(in_c=1, out_c=32).to("cuda")
optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True, fused=True)
# optimizer = AdamW(model.parameters(), lr=0.001)
# criterion = multi_task_loss
criterion = nn.MSELoss()
device = "cuda"
#%%

def calculate_sdri(true_percussion, predicted_percussion, mixture_audio):
    # Calculate SDR for mixture audio
    mixture_sdr = calculate_sdr(true_percussion, mixture_audio)
    # Calculate SDR for the predicted percussive signal
    predicted_sdr = calculate_sdr(true_percussion, predicted_percussion)
    # SDR Improvement is the difference between the predicted SDR and the mixture SDR
    sdri = predicted_sdr - mixture_sdr
    return sdri

def calculate_sdr(true_percussion, predicted_percussion):
    # Assuming true_percussion and predicted_percussion are tensors of the same shape
    noise = true_percussion - predicted_percussion
    sdr = 10 * torch.log10(torch.sum(true_percussion ** 2) / torch.sum(noise ** 2))
    return sdr.item()

def calculate_si_sdr(true_percussion, predicted_percussion):
    # Rescale predicted to match the scale of true percussive audio
    alpha = torch.sum(true_percussion * predicted_percussion) / torch.sum(predicted_percussion ** 2)
    true_scaled = alpha * predicted_percussion
    noise = true_percussion - true_scaled
    si_sdr = 10 * torch.log10(torch.sum(true_scaled ** 2) / torch.sum(noise ** 2))
    return si_sdr.item()


# %%
# Train the model

train_losses = []
val_losses = []
best_val_loss = np.inf
patience = 5
num_epochs = 30

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
        output_waveform = model(mixture)['waveform']

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
            output_waveform = model(mixture)['waveform']
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
                        filename='checkpoint_last_epoch_{}'.format(epoch))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 5
            # torch.save(model.state_dict(), 'best_model.pth')
            torch.save(model.state_dict(), 'best_model_last.pth')
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
#%%
# Load the best model
model.load_state_dict(torch.load('best_model_last.pth', weights_only=True))

#%%
# test without all the stft, mulitlabel etc, we only keep the names of the files, value of k 

model.eval()
test_loss = 0
percussion_files_l = []
noise_names_l = []
mix_names_l = []
k_values_l = []
classes_l = []
noise_audio_l = []

output_waveform_l = []
true_percussion_l = []
mixtures_l = []

test_bar = tqdm(test_loader, desc=f"Testing Loss: {
                test_loss:.4f}", colour='red')
with torch.no_grad():
    for i, batch in enumerate(test_bar):
        # Move data to device
        mixture = batch['mixture_audio'].to(device)
        # mix_stft = batch['mix_stft'].to(device)
        true_percussion = batch['percussion_audio'].to(device)
        noise_combined= batch['noise_audio']
        percussion_files = batch['percussion_file']
        noise_names = batch['noise_name']
        mix_names = batch['mix_name']
        k_values = batch['k']
        classes = batch['noise_classes']

        # Forward pass
        # output, class_output = model(torch.abs(mix_stft))
        output_waveform = model(mixture)
        output_waveform = output_waveform['waveform']

        # Calculate the loss
        loss = criterion(output_waveform, true_percussion)
        test_loss += loss.item()

        test_bar.set_description(
            f"Testing Loss: {test_loss/(i+1):.4f}")

        # concatenate inside the loop
        percussion_files_l.extend(percussion_files)
        noise_names_l.extend(noise_names)
        mix_names_l.extend(mix_names)
        k_values_l.extend(k_values)
        classes_l.extend(classes)
        noise_audio_l.extend(noise_combined)
        output_waveform_l.extend(output_waveform.cpu().numpy())
        true_percussion_l.extend(true_percussion.cpu().numpy())
        mixtures_l.extend(mixture.cpu().numpy())
        
    test_loss /= len(test_loader)
    
    # concatenate final values
    percussion_files_l.extend(percussion_files)
    noise_names_l.extend(noise_names)
    mix_names_l.extend(mix_names)
    k_values_l.extend(k_values)
    classes_l.extend(classes)
    noise_audio_l.extend(noise_combined)
    output_waveform_l.extend(output_waveform.cpu().numpy())
    true_percussion_l.extend(true_percussion.cpu().numpy())
    mixtures_l.extend(mixture.cpu().numpy())
    
    print(f"Testing Loss: {test_loss:.4f}")
    
    
#%%
# see shape of the lists
print(len(percussion_files_l))
print(len(noise_names_l))
print(len(mix_names_l))
print(len(k_values_l))
print(len(classes_l))
print(len(noise_audio_l))
print(len(output_waveform_l))
print(len(true_percussion_l))
print(len(mixtures_l))

#%%
# change the lists to numpy arrays
percussion_files = np.array(percussion_files_l)
noise_names = np.array(noise_names_l)
mix_names = np.array(mix_names_l)
k_values = np.array(k_values_l)
classes = np.array(classes_l)
noise_audio = np.array(noise_audio_l)
output_waveform = np.array(output_waveform_l)
true_percussion = np.array(true_percussion_l)
mixtures = np.array(mixtures_l)

#%%

# shape of all
arrays = [percussion_files, noise_names, mix_names, k_values, classes, noise_audio, output_waveform, true_percussion, mixtures]
for array in arrays:
    print(array.shape)

# (1494,)
# (1494,)
# (1494,)
# (1494,)
# (1494,)
# (1494, 1, 31248)
# (1494, 1, 31248)
# (1494, 1, 31248)
# (1494, 1, 31248)

#%%
# before plotting the true perc we need to apply the same preprocessing when we used them to create mix:
# - pad audio center
# - normalize loudness

#%%

# plot the waveform of the separated percussion with their respective noise, class and k value
# we can plot the waveform of the noise as well
for i in range(5):
    plt.figure(figsize=(10, 5))
    plt.subplot(4, 1, 1)
    plt.plot(mixtures[i,0])
    plt.title(f'Mixture {mix_names[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    
    plt.subplot(4, 1, 2)
    plt.plot(noise_audio[i,0]) # in the mix they are multiplied by (1-k) so we do the same here to compare
    plt.title(f'Noise {noise_names[i]}, Class {classes[i]}, (1-k) = {1-k_values[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    
    plt.subplot(4, 1, 3)
    plt.plot(true_percussion[i,0]) # in the mix they are multiplied by k so we do the same here to compare
    plt.title(f'True Percussion {percussion_files[i]}, k = {k_values[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    
    plt.subplot(4, 1, 4)
    plt.plot(output_waveform[i,0])
    plt.title(f'Separated Percussion {percussion_files[i]}')
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    
    plt.tight_layout()
    plt.show()

#%%

def get_stft(audio):
    stft = torch.stft(torch.tensor(audio, dtype=torch.float32).to("cuda"), n_fft=256, hop_length=64, win_length=256, window=torch.hann_window(window_length=256, device='cuda'), return_complex=True)
    return stft

# compute their spectrogram
def plot_spectrogram_alone(audio, title):
    stft = get_stft(audio)
    mag = torch.abs(stft)
    
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(), ref=np.max), y_axis='linear', x_axis='time', sr=7812)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

#%%    
# plot the spectrogram of the separated percussion with their respective noise, class and k value
# we can plot the spectrogram of the noise as well

for i in range(5):
    plot_spectrogram_alone(mixtures[i], f'Mixture {mix_names[i]}')

    plot_spectrogram_alone(noise_audio[i], f'Noise {noise_names[i]}, Class {classes[i]}, (1-k) = {1-k_values[i]}')
    
    plot_spectrogram_alone(true_percussion[i], f'True Percussion {percussion_files[i]}, k = {k_values[i]}')
    
    plot_spectrogram_alone(output_waveform[i], f'Separated Percussion {percussion_files[i]}')
        
    
    
#%%
    

def plot_audio_spectrogram(audio, title):
    stft = get_stft(audio)
    mag = torch.abs(stft)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(audio[0])
    plt.title(title)
    plt.ylim(-1, 1)  # Set the y-axis limits to -1 and 1
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(mag[0].cpu().numpy(), ref=np.max), y_axis='linear', x_axis='time', sr=7812)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    
#%%
# now plot all audios with their spectrograms, with audio at the top and spectrogram at the bottom

for i in range(5):
    plot_audio_spectrogram(mixtures[i], f'Mixture {mix_names[i]}')
    plot_audio_spectrogram(noise_audio[i], f'Noise {noise_names[i]}, Class {classes[i]}, (1-k) = {1-k_values[i]}')
    plot_audio_spectrogram(true_percussion[i], f'True Percussion {percussion_files[i]}, k = {k_values[i]}')
    plot_audio_spectrogram(output_waveform[i], f'Separated Percussion {percussion_files[i]}')

# %%

