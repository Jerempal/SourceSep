# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary as summary
from torchlibrosa.stft import STFT, ISTFT, magphase
from base import Base
import numpy as np
from typing import Tuple, Dict, NoReturn

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
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


class ResUNetv1(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResUNetv1, self).__init__()

        """ Encoder 1 """
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_c, out_c,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c,
                      kernel_size=3, padding=1),
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
        self.last_layer = nn.Sequential(
            nn.Conv2d(out_c, 3, kernel_size=1, padding=0),
        )

    def forward(self, inputs):
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
        output = self.last_layer(decoder3)

        return output


# %%
model = ResUNetv1(in_c=1, out_c=32).to("cuda")


summary.summary(model, (1, 129, 489))


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
        # if x.shape[2:] != skip.shape[2:]:
        #     x = F.interpolate(
        #         x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)

        # Concatenate
        x = torch.cat([x, skip], dim=1)

        # Residual block
        x = self.residual_block(x)

        return x


class ResUNetv2(nn.Module, Base):
    def __init__(self, in_c, out_c):
        super(ResUNetv2, self).__init__()

        window_size = 256
        hop_size = 64
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.output_channels = 3
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
        self.last_layer = nn.Sequential(
            # nn.Conv2d(out_c, 1, kernel_size=1, padding='same'),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )

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
        print(f'batch_size: {batch_size}')
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
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos +
            cos_in[:, None, :, :, :] * mask_sin
        )

        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)

        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
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
        print(f'mixtures shape',
              mixtures.shape)  # why torch.Size([2, 1, 31248]) ????
        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag
        print(f'x shape before pad', x.shape)
        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        print(f'origin_len', origin_len)
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)
                ) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))

        """(batch_size, channels, padded_time_steps, freq_bins)"""
        print(f'x shape', x.shape)
        # Let frequency bins be evenly divided by 2, e.g., 489 -> 488.
        x = x[..., 0: x.shape[-1] - 1]  # (bs, channels, T, F)
        # UNet
        print(f'x shape', x.shape)
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
        print(f'x shape', x.shape)
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


model = ResUNetv2(in_c=1, out_c=32).to("cuda")
summary.summary(model, (1, 31248))


# %%
