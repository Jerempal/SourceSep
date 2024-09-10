#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio

def SDR(target, prediction):
    """
        Compute the signal-to-distortion ratio for the target and prediction signals (audio).

    Args:
        target (tensor): true percussion signal
        prediction (tensor): predicted percussion signal

    Returns:
        _type_: signal-to-distortion ratio
    """
    # Calculate the signal-to-distortion ratio

    target_power = torch.sum(target**2)  # target is the percussion
    error_power = torch.sum((target - prediction)**2)

    sdr = 10 * torch.log10(target_power / error_power)
    return sdr

def SISDR(target, prediction):
    """
    Calculate the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) between the target and prediction signals.

    Parameters:
    - target (torch.Tensor): The target source signal.
    - prediction (torch.Tensor): The predicted signal.

    Returns:
    - SDRi (float): The SDR improvement in dB.
    """
    # Ensure input tensors are of the same shape
    if target.shape != prediction.shape:
        raise ValueError(
            "Target and prediction tensors must have the same shape")

    # Compute the dot product (s_hat.T @ s)
    dot_product = torch.dot(prediction.flatten(), target.flatten())

    # Compute the norm squared of the target signal
    target_norm_squared = torch.norm(target) ** 2

    # Compute the scaling factor
    scaling_factor = dot_product / target_norm_squared

    # Compute the scaled predicted signal
    scaled_prediction = scaling_factor * target

    # Compute the error signal
    error_signal = scaled_prediction - prediction

    # Compute the numerator and denominator for SDRi
    numerator = torch.norm(scaled_prediction) ** 2
    denominator = torch.norm(error_signal) ** 2

    # Compute SDR improvement
    si_sdr = 10 * torch.log10(numerator / denominator)

    return si_sdr

def SDR_i(target, prediction, mixture):
    """
        Compute the signal-to-distortion ratio improvement

    Args:
        target (_type_): true percussion signal
        prediction (_type_): predicted percussion signal
        mixture (_type_): mixture of the percussion and noise signals

    Returns:
        _type_: signal-to-distortion ratio improvement
    """
    # Calculate the signal-to-distortion ratio improvement

    sdr = SDR(target, prediction)
    sdr_mixture = SDR(target, mixture)
    sdr_improvement = sdr - sdr_mixture

    return sdr_improvement

#%%
sdr_metric = SignalDistortionRatio()
si_sdr_metric = ScaleInvariantSignalDistortionRatio()

val_loss = 0
val_sdr_epoch = 0
val_si_sdr_epoch = 0

pred = torch.rand(25, 1, 31248)
target = torch.rand(25, 1, 31248)
mixture = torch.rand(25, 1, 31248)

# Calculate SDR and SI-SDR using torchmetrics
si_sdr = si_sdr_metric(pred, target)
sdri = sdr_metric(target, pred) - \
    sdr_metric(target, mixture)

val_sdr_epoch += sdri.item()
val_si_sdr_epoch += si_sdr.item()

print(f"SDR improvement: {val_sdr_epoch}")
print(f"SI-SDR: {val_si_sdr_epoch}")
print(f'si_sdr: {si_sdr}')
print(f'sdri: {sdri}')

#%%
# define the loss function


def spectral_loss(prediction, target):
    # spectral loss
    # log magnitude spectrogram loss
    # target is the percussion target_stft we take the magnitude of the target hence the abs
    # prediction is the predicted percussion stft
    # ||log(abs(target) + 1e-9) - log(abs(prediction) + 1e-9)||_L1, L1 norm

    # Calculate the log magnitude spectrogram loss

    target_stft = torch.log(torch.abs(target) + 1e-9)
    prediction_stft = torch.log(torch.abs(prediction) + 1e-9)
    loss = nn.L1Loss()(target_stft, prediction_stft)

    return loss


def loss_wav(prediction, target):
    # waveform loss
    # ||target - prediction||_L1, L1 norm

    # Calculate the waveform loss

    loss = F.l1_loss(target, prediction)
    
    return loss


def loss_mse(prediction, target):
    # mean squared error loss
    # ||target - prediction||_L2, L2 norm

    # Calculate the mean squared error loss

    loss = F.mse_loss(target, prediction)

    return loss
