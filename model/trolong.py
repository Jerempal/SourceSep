

# class ConvBlockRes(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: Tuple,
#         momentum: float,
#     ):
#         r"""Residual block."""
#         super(ConvBlockRes, self).__init__()

#         padding = [kernel_size[0] // 2, kernel_size[1] // 2]

#         self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
#         self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

#         self.conv1 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=(1, 1),
#             dilation=(1, 1),
#             padding=padding,
#             bias=False,
#         )

#         self.conv2 = nn.Conv2d(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=(1, 1),
#             dilation=(1, 1),
#             padding=padding,
#             bias=False,
#         )

#         if in_channels != out_channels:
#             self.shortcut = nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=(1, 1),
#                 stride=(1, 1),
#                 padding=(0, 0),
#             )
#             self.is_shortcut = True
#         else:
#             self.is_shortcut = False

#     def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
#         r"""Forward data into the module.

#         Args:
#             input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

#         Returns:
#             output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
#         """

#         x = self.conv1(F.leaky_relu_(
#             self.bn1(input_tensor), negative_slope=0.01))
#         x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))

#         if self.is_shortcut:
#             return self.shortcut(input_tensor) + x
#         else:
#             return input_tensor + x


# class EncoderBlockRes1B(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: Tuple,
#         downsample: Tuple,
#         momentum: float,
#     ):
#         r"""Encoder block, contains 8 convolutional layers."""
#         super(EncoderBlockRes1B, self).__init__()

#         self.conv_block1 = ConvBlockRes(
#             in_channels, out_channels, kernel_size, momentum,
#         )
#         self.downsample = downsample

#     def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
#         r"""Forward data into the module.

#         Args:
#             input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

#         Returns:
#             encoder_pool: (batch_size, output_feature_maps, downsampled_time_steps, downsampled_freq_bins)
#             encoder: (batch_size, output_feature_maps, time_steps, freq_bins)
#         """
#         encoder = self.conv_block1(input_tensor)
#         encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
#         return encoder_pool, encoder


# class DecoderBlockRes1B(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: Tuple,
#         upsample: Tuple,
#         momentum: float,
#     ):
#         r"""Decoder block, contains 1 transposed convolutional and 8 convolutional layers."""
#         super(DecoderBlockRes1B, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = upsample

#         self.conv1 = torch.nn.ConvTranspose2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=self.stride,
#             stride=self.stride,
#             padding=(0, 0),
#             bias=False,
#             dilation=(1, 1),
#         )

#         self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
#         self.conv_block2 = ConvBlockRes(
#             out_channels * 2, out_channels, kernel_size, momentum,
#         )
#         self.bn2 = nn.BatchNorm2d(in_channels, momentum=momentum)

#     def forward(
#         self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor
#     ) -> torch.Tensor:
#         r"""Forward data into the module.

#         Args:
#             input_tensor: (batch_size, input_feature_maps, downsampled_time_steps, downsampled_freq_bins)
#             concat_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

#         Returns:
#             output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
#         """

#         x = self.conv1(F.leaky_relu_(self.bn1(input_tensor)))
#         # (batch_size, input_feature_maps, time_steps, freq_bins)

#         x = torch.cat((x, concat_tensor), dim=1)
#         # (batch_size, input_feature_maps * 2, time_steps, freq_bins)

#         x = self.conv_block2(x)
#         # output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)

#         return x


# class ResUNet30_Base(nn.Module, Base):
#     def __init__(self, input_channels, output_channels):
#         super(ResUNet30_Base, self).__init__()

#         window_size = 256
#         hop_size = 64
#         center = True
#         pad_mode = "reflect"
#         window = "hann"
#         momentum = 0.01

#         self.output_channels = output_channels
#         self.target_sources_num = 1
#         self.K = 3

#         # This number equals 2^{#encoder_blcoks}
#         self.time_downsample_ratio = 2 ** 5

#         self.stft = STFT(
#             n_fft=window_size,
#             hop_length=hop_size,
#             win_length=window_size,
#             window=window,
#             center=center,
#             pad_mode=pad_mode,
#             freeze_parameters=True,
#         )

#         self.istft = ISTFT(
#             n_fft=window_size,
#             hop_length=hop_size,
#             win_length=window_size,
#             window=window,
#             center=center,
#             pad_mode=pad_mode,
#             freeze_parameters=True,
#         )

#         self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

#         self.pre_conv = nn.Conv2d(
#             in_channels=input_channels,
#             out_channels=32,
#             kernel_size=(1, 1),
#             stride=(1, 1),
#             padding=(0, 0),
#             bias=True,
#         )

#         self.encoder_block1 = EncoderBlockRes1B(
#             in_channels=32,
#             out_channels=32,
#             kernel_size=(3, 3),
#             downsample=(2, 2),
#             momentum=momentum,
#         )
#         self.encoder_block2 = EncoderBlockRes1B(
#             in_channels=32,
#             out_channels=64,
#             kernel_size=(3, 3),
#             downsample=(2, 2),
#             momentum=momentum,
#         )
#         self.encoder_block3 = EncoderBlockRes1B(
#             in_channels=64,
#             out_channels=128,
#             kernel_size=(3, 3),
#             downsample=(2, 2),
#             momentum=momentum,
#         )
#         self.encoder_block4 = EncoderBlockRes1B(
#             in_channels=128,
#             out_channels=256,
#             kernel_size=(3, 3),
#             downsample=(2, 2),
#             momentum=momentum,
#         )
#         self.encoder_block5 = EncoderBlockRes1B(
#             in_channels=256,
#             out_channels=384,
#             kernel_size=(3, 3),
#             downsample=(2, 2),
#             momentum=momentum,
#         )
#         self.encoder_block6 = EncoderBlockRes1B(
#             in_channels=384,
#             out_channels=384,
#             kernel_size=(3, 3),
#             downsample=(1, 2),
#             momentum=momentum,
#         )
#         self.conv_block7a = EncoderBlockRes1B(
#             in_channels=384,
#             out_channels=384,
#             kernel_size=(3, 3),
#             downsample=(1, 1),
#             momentum=momentum,
#         )
#         self.decoder_block1 = DecoderBlockRes1B(
#             in_channels=384,
#             out_channels=384,
#             kernel_size=(3, 3),
#             upsample=(1, 2),
#             momentum=momentum,
#         )
#         self.decoder_block2 = DecoderBlockRes1B(
#             in_channels=384,
#             out_channels=384,
#             kernel_size=(3, 3),
#             upsample=(2, 2),
#             momentum=momentum,
#         )
#         self.decoder_block3 = DecoderBlockRes1B(
#             in_channels=384,
#             out_channels=256,
#             kernel_size=(3, 3),
#             upsample=(2, 2),
#             momentum=momentum,
#         )
#         self.decoder_block4 = DecoderBlockRes1B(
#             in_channels=256,
#             out_channels=128,
#             kernel_size=(3, 3),
#             upsample=(2, 2),
#             momentum=momentum,
#         )
#         self.decoder_block5 = DecoderBlockRes1B(
#             in_channels=128,
#             out_channels=64,
#             kernel_size=(3, 3),
#             upsample=(2, 2),
#             momentum=momentum,
#         )
#         self.decoder_block6 = DecoderBlockRes1B(
#             in_channels=64,
#             out_channels=32,
#             kernel_size=(3, 3),
#             upsample=(2, 2),
#             momentum=momentum,
#         )

#         self.after_conv = nn.Conv2d(
#             in_channels=32,
#             out_channels=output_channels * self.K,
#             kernel_size=(1, 1),
#             stride=(1, 1),
#             padding=(0, 0),
#             bias=True,
#         )

#     def feature_maps_to_wav(
#         self,
#         input_tensor: torch.Tensor,
#         sp: torch.Tensor,
#         sin_in: torch.Tensor,
#         cos_in: torch.Tensor,
#         audio_length: int,
#     ) -> torch.Tensor:
#         r"""Convert feature maps to waveform.

#         Args:
#             input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
#             sp: (batch_size, input_channels, time_steps, freq_bins)
#             sin_in: (batch_size, input_channels, time_steps, freq_bins)
#             cos_in: (batch_size, input_channels, time_steps, freq_bins)

#             (There is input_channels == output_channels for the source separation task.)

#         Outputs:
#             waveform: (batch_size, target_sources_num * output_channels, segment_samples)
#         """
#         batch_size, _, time_steps, freq_bins = input_tensor.shape

#         x = input_tensor.reshape(
#             batch_size,
#             self.target_sources_num,
#             self.output_channels,
#             self.K,
#             time_steps,
#             freq_bins,
#         )
#         # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

#         mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
#         _mask_real = torch.tanh(x[:, :, :, 1, :, :])
#         _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
#         # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
#         _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
#         # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

#         # Y = |Y|cos∠Y + j|Y|sin∠Y
#         #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
#         #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
#         out_cos = (
#             cos_in[:, None, :, :, :] * mask_cos -
#             sin_in[:, None, :, :, :] * mask_sin
#         )
#         out_sin = (
#             sin_in[:, None, :, :, :] * mask_cos +
#             cos_in[:, None, :, :, :] * mask_sin
#         )
#         # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
#         # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

#         # Calculate |Y|.
#         out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
#         # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
#         # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

#         # Calculate Y_{real} and Y_{imag} for ISTFT.
#         out_real = out_mag * out_cos
#         out_imag = out_mag * out_sin
#         # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

#         # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
#         # N = batch_size * target_sources_num * output_channels
#         shape = (
#             batch_size * self.target_sources_num * self.output_channels,
#             1,
#             time_steps,
#             freq_bins,
#         )
#         out_real = out_real.reshape(shape)
#         out_imag = out_imag.reshape(shape)

#         # ISTFT.
#         x = self.istft(out_real, out_imag, audio_length)
#         # (batch_size * target_sources_num * output_channels, segments_num)

#         # Reshape.
#         waveform = x.reshape(
#             batch_size, self.target_sources_num * self.output_channels, audio_length
#         )
#         # (batch_size, target_sources_num * output_channels, segments_num)

#         return waveform

#     def forward(self, mixtures):
#         """
#         Args:
#           input: (batch_size, segment_samples, channels_num)

#         Outputs:
#           output_dict: {
#             'wav': (batch_size, segment_samples, channels_num),
#             'sp': (batch_size, channels_num, time_steps, freq_bins)}
#         """

#         mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
#         x = mag
#         # Batch normalization
#         x = x.transpose(1, 3)
#         x = self.bn0(x)
#         x = x.transpose(1, 3)
#         """(batch_size, chanenls, time_steps, freq_bins)"""

#         # Pad spectrogram to be evenly divided by downsample ratio.
#         origin_len = x.shape[2]
#         pad_len = (
#             int(np.ceil(x.shape[2] / self.time_downsample_ratio)
#                 ) * self.time_downsample_ratio
#             - origin_len
#         )
#         x = F.pad(x, pad=(0, 0, 0, pad_len))
#         """(batch_size, channels, padded_time_steps, freq_bins)"""
#         # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
#         x = x[..., 0: x.shape[-1] - 1]  # (bs, channels, T, F)
#         # UNet
#         x = self.pre_conv(x)
#         # x1_pool: (bs, 32, T / 2, F / 2)
#         x1_pool, x1 = self.encoder_block1(x)
#         # x2_pool: (bs, 64, T / 4, F / 4)
#         x2_pool, x2 = self.encoder_block2(x1_pool)
#         # x3_pool: (bs, 128, T / 8, F / 8)
#         x3_pool, x3 = self.encoder_block3(x2_pool)
#         # x4_pool: (bs, 256, T / 16, F / 16)
#         x4_pool, x4 = self.encoder_block4(x3_pool)
#         # x5_pool: (bs, 384, T / 32, F / 32)
#         x5_pool, x5 = self.encoder_block5(x4_pool)
#         # x6_pool: (bs, 384, T / 32, F / 64)
#         x6_pool, x6 = self.encoder_block6(x5_pool)
#         x_center, _ = self.conv_block7a(x6_pool)  # (bs, 384, T / 32, F / 64)
#         # (bs, 384, T / 32, F / 32)
#         x7 = self.decoder_block1(x_center, x6)
#         # (bs, 384, T / 16, F / 16)
#         x8 = self.decoder_block2(x7, x5)
#         # (bs, 256, T / 8, F / 8)
#         x9 = self.decoder_block3(x8, x4)
#         # (bs, 128, T / 4, F / 4)
#         x10 = self.decoder_block4(x9, x3)
#         # (bs, 64, T / 2, F / 2)
#         x11 = self.decoder_block5(x10, x2)
#         x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)

#         x = self.after_conv(x12)

#         # Recover shape
#         x = F.pad(x, pad=(0, 1))
#         x = x[:, :, 0:origin_len, :]

#         audio_length = mixtures.shape[2]

#         # Recover each subband spectrograms to subband waveforms. Then synthesis
#         # the subband waveforms to a waveform.
#         separated_audio = self.feature_maps_to_wav(
#             input_tensor=x,
#             # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
#             sp=mag,
#             # sp: (batch_size, input_channels, T, F')
#             sin_in=sin_in,
#             # sin_in: (batch_size, input_channels, T, F')
#             cos_in=cos_in,
#             # cos_in: (batch_size, input_channels, T, F')
#             audio_length=audio_length,
#         )
#         # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)

#         output_dict = {'waveform': separated_audio}

#         return output_dict


# model = ResUNet30_Base(input_channels=1, output_channels=1).to("cuda")
# summary.summary(model, (1, 31248))

# #
