import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
import os
import random
import torchaudio

n_fft = 256
hop_length = n_fft // 4


def get_stft(audio):
    stft = torch.stft(torch.tensor(audio, dtype=torch.float32).to("cuda"), n_fft=256, hop_length=64,
                      win_length=256, window=torch.hann_window(window_length=256, device='cuda'), return_complex=True)
    return stft


def create_mixture(percussion_audio, noise_audio, k):

    percussion_audio /= np.max(np.abs(percussion_audio))
    noise_audio /= np.max(np.abs(noise_audio))

    # Mix audio files
    mixture_audio = k*percussion_audio + (1-k)*noise_audio

    # Calculate stft
    mix_stft = get_stft(mixture_audio)

    return mixture_audio, mix_stft


def audio_from_spectrogram(mag, phase):
    stft = mag * torch.exp(1j * phase)
    audio = torch.istft(stft, n_fft=n_fft, hop_length=hop_length,
                        length=4 * 7812, window=torch.hann_window(n_fft, device='cuda'))
    return audio


def pad_audio_center(audio_path, sample_rate=7812, target_length=31248):
    audio, sr = librosa.load(audio_path, sr=sample_rate)

    if len(audio) < target_length:
        pad_len = (target_length - len(audio)) // 2
        audio = np.pad(audio, (pad_len, target_length -
                       len(audio) - pad_len), 'constant')
    
    audio = audio[:target_length]
    # audio = torch.tensor(audio, dtype=torch.float32)
    
    return audio



# def pad_audio_center(audio, target_length=31248):

#     if len(audio) < target_length:
#         pad_len = (target_length - len(audio)) // 2
#         audio = np.pad(audio, (pad_len, target_length -
#                        len(audio) - pad_len), 'constant')
#     return audio[:target_length]
torchaudio.transforms.Vol

def dynamic_loudnorm(audio, reference, lower_db=-6, higher_db=6):
    # Rescale the audio to match the loudness of the reference (percussion)
    rescaled_audio = rescale_to_match_energy(audio, reference)
    delta_loudness = random.randint(
        lower_db, higher_db)  # Random loudness variation
    gain = np.power(10.0, delta_loudness / 20.0)
    # gain = np.power(10.0, -3 / 20.0)
    rescaled_audio *= gain
    return rescaled_audio

def rescale_to_match_energy(segment1, segment2):
    ratio = get_energy_ratio(segment1, segment2)
    return segment1 / ratio


def get_energy(x):
    return torch.mean(x ** 2)
    # return np.mean(x ** 2)


def get_energy_ratio(segment1, segment2):
    energy1 = get_energy(segment1)
    # Prevent division by zero
    energy2 = max(get_energy(segment2), 1e-10)
    ratio = (energy1 / energy2) ** 0.5
    return torch.clamp(ratio, 0.02, 50)  # Avoid extreme scaling
    # return np.clip(ratio, 0.02, 50)  # Avoid extreme scaling

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir='checkpoints', filename='checkpoint.pth'):
    # Create the directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, checkpoint_path)
    print(f"Checkpoint saved at '{checkpoint_path}'")


def load_checkpoint(model, optimizer, checkpoint_dir='checkpoints', filename='checkpoint.pth'):
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    print(f"Checkpoint loaded from '{checkpoint_path}'")

    return model, optimizer, epoch, train_loss, val_loss
