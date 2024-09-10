# from torchviz import make_dot
import torchsummary as summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import Base
from torchlibrosa.stft import magphase, STFT, ISTFT


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


# class ResUNet(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(ResUNet, self).__init__()

#         """ Encoder 1 """
#         self.encoder_block1 = nn.Sequential(
#             nn.Conv2d(in_c, out_c,
#                       kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(out_c, out_c,
#                       kernel_size=3, stride=1, padding=1),
#         )

#         """ Shortcut Connection """
#         self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

#         """ Encoder 2 and 3"""
#         self.encoder_block2 = ResidualBlock(
#             out_c, out_c * 2, stride=2)
#         self.encoder_block3 = ResidualBlock(
#             out_c * 2, out_c * 4, stride=2)

#         """ Bridge """
#         self.bridge = ResidualBlock(
#             out_c * 4, out_c * 8, stride=2)

#         """ Decoder """
#         self.decoder_block1 = DecoderBlock(out_c * 8, out_c * 4)
#         self.decoder_block2 = DecoderBlock(out_c * 4, out_c * 2)
#         self.decoder_block3 = DecoderBlock(out_c * 2, out_c)

#         """ Output """
#         self.output = nn.Sequential(
#             nn.Conv2d(out_c, 3, kernel_size=1, padding=0),
#         )

#         # self.classifier = nn.Sequential(
#         #     nn.MaxPool2d(kernel_size=2, stride=2),
#         #     nn.ReLU(),
#         #     nn.Conv2d(out_c*4, out_c*2, kernel_size=3, padding=1),
#         #     nn.MaxPool2d(kernel_size=2, stride=2),
#         #     nn.ReLU(),

#         #     # Linear
#         #     nn.Flatten(),
#         #     nn.Linear(out_c*2 * 8 * 30, out_c*2),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.3),
#         #     nn.Linear(out_c*2, out_c),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.3),
#         #     nn.Linear(out_c, 8)
#         # )

#         self.classifier = nn.Sequential(
#             nn.MaxPool2d(4, 4),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(out_c*4, out_c*2, kernel_size=3, padding=1),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),

#             nn.Flatten(),
#             # nn.Linear(out_c*4 * 16 * 61, 128),
#             # nn.Linear(out_c*4 * 8 * 30, 128),
#             nn.Linear(out_c*2 * 4 * 15, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 8)
#         )

#     def forward(self, x):

#         x = x.unsqueeze(1)

#         """ Encoder 1 """
#         encoder1 = self.encoder_block1(x)
#         s = self.shortcut(x)
#         skip1 = encoder1 + s

#         """ Encoder 2 and 3 """
#         skip2 = self.encoder_block2(skip1)
#         skip3 = self.encoder_block3(skip2)

#         """ Bridge """
#         bridge = self.bridge(skip3)

#         """ Decoder """
#         decoder1 = self.decoder_block1(bridge, skip3)
#         decoder2 = self.decoder_block2(decoder1, skip2)
#         decoder3 = self.decoder_block3(decoder2, skip1)

#         """ Output """
#         output = self.output(decoder3)

#         output_masks_dict = {
#             'mag_mask': torch.sigmoid(output[:, 0, :, :]),
#             'real_mask': torch.tanh(output[:, 1, :, :]),
#             'imag_mask': torch.tanh(output[:, 2, :, :])
#         }

#         class_output = self.classifier(skip3)
#         # return output, class_output

#         return output_masks_dict, class_output

class SepModel(nn.Module, Base):
    def __init__(self, in_c, out_c):
        super(SepModel, self).__init__()

        self.output_channels = 1
        self.K = 3

        # downsample ratio
        self.time_downsample_ratio = 2**3

        self.stft = STFT(
            n_fft=256,
            hop_length=64,
            win_length=256,
            window='hann',
            center=True,
            pad_mode='reflect',
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=256,
            hop_length=64,
            win_length=256,
            window='hann',
            center=True,
            pad_mode='reflect',
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
            input_tensor: (batch_size, output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, 2, :, :])

        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, output_channels, time_steps, freq_bins)

        # Y = |Y|exp(j∠Y) = |Y|exp(j(∠X + ∠M))
        #   = |Y|(cos∠Y + jsin∠Y)
        #   = |Y|cos∠Y + j|Y|sin∠Y
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
        # out_cos: (batch_size, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, :, :, :] * mask_mag)
        # out_mag: (batch_size, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size *  output_channels
        # shape = (batch_size * self.
        #  self.output_channels, 1, time_steps, freq_bins)

        shape = (batch_size * self.output_channels, 1, time_steps, freq_bins)
        shape = (batch_size, 1, time_steps, freq_bins)

        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size *  output_channels, segments_num)

        # Reshape.
        # waveform = x.reshape(batch_size,self.output_channels, audio_length)
        waveform = x.reshape(batch_size, self.output_channels, audio_length)
        # (batch_size,  output_channels, segments_num)
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
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)
                ) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len)) 

        """(batch_size, channels, padded_time_steps, freq_bins)"""
        # les piques de fréquences doivent être divisé par 2, e.g., 489 -> 488.

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
        x = self.after_conv(decoder3)

        x = F.pad(x, pad=(0, 1)) 
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
        )

        output_dict = {'waveform': separated_audio}

        return output_dict


# # %%
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SepModel(1, 32).to(device)
# summary.summary(model, (1, 31248), batch_size=32)
# # %%
# # torchviz

# import torch
# from torchviz import make_dot

# # Generate a random input tensor
# x = torch.randn(1, 1, 31248).to(device)

# # Forward pass through the model
# output = model(x)

# # If the output is a dictionary, extract the relevant tensor
# if isinstance(output, dict):
#     # Extract the tensor using the correct key 'waveform'
#     output = output['waveform']

# # Create a dictionary of model parameters
# params = {name: param for name, param in model.named_parameters()}

# # Visualize the model
# make_dot(output, params=params)

# # %%
# # using

