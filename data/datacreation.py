import numpy
import os
import pandas as pd
import torch
import pyloudnorm as pyln
import random
import soundfile as sf
from tqdm import tqdm
import librosa
from config import DATASET_PERCUSSION_PATH, DATASET_NOISE_PATH, DATASET_MIX_AUDIO_PATH, DATASET_MIX_AUDIO_PATH2
from data.utils import dynamic_loudnorm
from multiprocessing import Pool, cpu_count


# Setup
random.seed(42)  # Ensure reproducibility

# Loudness meter
sample_rate = 7812
meter = pyln.Meter(sample_rate)

# Noise classes
noise_class_list = [
    'air_conditioner',
    'car_horn',
    'children_playing',
    'dog_bark',
    'drilling',
    'engine_idling',
    'siren',
    'jackhammer'
]

def pad_audio_center(audio_path, sample_rate=7812, target_length=31248):
    audio, sr = librosa.load(audio_path, sr=sample_rate)

    if len(audio) < target_length:
        pad_len = (target_length - len(audio)) // 2
        audio = numpy.pad(audio, (pad_len, target_length -
                       len(audio) - pad_len), 'constant')
    
    audio = audio[:target_length]
    
    return audio

# Helper functions for normalization and mixing
def normalize_loudness(audio, target_loudness=-3):
    meter = pyln.Meter(7812)
    loudness = meter.integrated_loudness(audio)
    if loudness == -float('inf'):
        audio /= numpy.max(numpy.abs(audio))
        loudness = meter.integrated_loudness(audio)

    audio = pyln.normalize.loudness(audio, loudness, target_loudness)

    # Ensure audio is within [-1, 1]
    max_amplitude = max(abs(audio))
    audio = audio / max_amplitude
    return torch.tensor(audio, dtype=torch.float32)


def create_mixture(percussion_audio, noise_audio, k):
    percussion_audio = k * percussion_audio
    noise_audio = (1 - k) * noise_audio
    mixture_audio = percussion_audio + noise_audio
    
    mixture_audio /= torch.max(torch.abs(mixture_audio))
    return mixture_audio, noise_audio

def create_dataset(metadata_noise, output_dir, num_mixes=7358, target_loudness=-3, max_noise_classes=2, k_values=[0.1, 0.2, 0.3, 0.4]):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    unique_id_counter = 0
    percussion_files = sorted(os.listdir(DATASET_PERCUSSION_PATH))
    percussion_files = [f for f in percussion_files if f.endswith('.wav')]
    total_percussion = len(percussion_files)
    
    # Shuffling and iterating through all noise files
    noise_files = metadata_noise.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculating how many mixes per percussion file
    mixes_per_percussion = num_mixes // total_percussion
    remainder_mixes = num_mixes % total_percussion
    
    noise_idx = 0
    for perc_idx, perc_file in enumerate(tqdm(percussion_files, desc="Creating Mixtures", colour='green')):
        percussion_path = os.path.join(DATASET_PERCUSSION_PATH, perc_file)
        percussion_audio = pad_audio_center(percussion_path)
        percussion_audio = normalize_loudness(percussion_audio, target_loudness)

        # Adjusting number of mixes for the current percussion file
        current_num_mixes = mixes_per_percussion + (1 if perc_idx < remainder_mixes else 0)
        
        for _ in range(current_num_mixes):
            num_noise_classes = random.randint(1, max_noise_classes)
            noise_classes = []
            k = random.choice(k_values)
            noise_audio_combined = torch.zeros_like(percussion_audio)

            selected_noise_files = []
            selected_noise_classes = []
            
            for _ in range(num_noise_classes):
                noise_row = noise_files.iloc[noise_idx]
                noise_idx = (noise_idx + 1) % len(noise_files)
                
                noise_file = os.path.join(DATASET_NOISE_PATH, f"fold{noise_row['fold']}", noise_row['slice_file_name'])
                noise_audio = pad_audio_center(noise_file)
                noise_audio = normalize_loudness(noise_audio, target_loudness)
                
                noise_audio_combined += dynamic_loudnorm(noise_audio, percussion_audio)
                selected_noise_files.append(noise_row['slice_file_name'])
                selected_noise_classes.append(noise_row['class'])
                
                noise_classes.append(noise_row['class'])

            noise_audio_combined = dynamic_loudnorm(noise_audio_combined, percussion_audio)
            noise_audio_combined /= torch.max(torch.abs(noise_audio_combined))
            mixture_audio, noise_audio_combined = create_mixture(percussion_audio, noise_audio_combined, k)
            
            unique_id_counter += 1
            mix_file_name = f"mixture_{perc_idx}_noise_{'_'.join(noise_classes)}_k_{k:.2f}_{unique_id_counter}.wav"
            mix_file_path = os.path.join(output_dir, mix_file_name)
            sf.write(mix_file_path, mixture_audio.cpu().numpy(), sample_rate)

            noise_file_name = f"noise_{perc_idx}_noise_{'_'.join(noise_classes)}_k_{k:.2f}_{unique_id_counter}.wav"
            noise_file_path = os.path.join(output_dir, noise_file_name)
            sf.write(noise_file_path, noise_audio_combined.cpu().numpy(), sample_rate)
            
            metadata.append({
                'percussion_file': perc_file,
                'mix_file': mix_file_name,
                'noise_files': ','.join(selected_noise_files),
                'noise_file': noise_file_name,
                'noise_classes': ','.join(selected_noise_classes),
                'k': k
            })

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"Dataset and metadata saved in '{output_dir}'")

if __name__ == "__main__":
    # Load noise metadata
    metadata_noise_path = os.path.join(DATASET_NOISE_PATH, "UrbanSound8k.csv")
    metadata_noise = pd.read_csv(metadata_noise_path)

    # Change classID 8 to 6
    metadata_noise['classID'] = metadata_noise['classID'].replace(8, 6)
    # Sort by class ID and fold and reset index
    metadata_noise = metadata_noise.sort_values(
        by=['classID', 'fold']).reset_index(drop=True)

    # Create dataset with mixtures and save metadata
    # output_dir = DATASET_MIX_AUDIO_PATH
    output_dir = DATASET_MIX_AUDIO_PATH2
    
    # Use multiprocessing to speed up the data creation process
    with Pool(cpu_count()) as p:
        p.starmap(create_dataset, [(metadata_noise, output_dir)])
    
    print("Data creation process completed")

output_dir = DATASET_MIX_AUDIO_PATH
output_dir2 = DATASET_MIX_AUDIO_PATH2
# Load metadata
metadata_path = os.path.join(output_dir, "metadata.csv")
metadata = pd.read_csv(metadata_path)

metadata_path = os.path.join(output_dir2, "metadata.csv")
metadata2 = pd.read_csv(metadata_path)

print(f"The number of unique noise classes: {metadata['noise_classes'].nunique()}")
print(f"The count of each k value: {metadata['k'].value_counts()}")
print(f"The number of unique percussion files: {metadata['percussion_file'].nunique()}")
print(f"The number of unique mix files: {metadata['mix_file'].nunique()}")
print(f"The number of unique noise files: {metadata['noise_files'].nunique()}")
print(f"The count of each noise file: {metadata['noise_files'].value_counts()}")
print(metadata.groupby("noise_files").size().describe())

# see the different noise files saved
noise_files = metadata['noise_files'].str.split(',')
noise_files = [item for sublist in noise_files for item in sublist]
print(f"The number of unique noise files: {len(set(noise_files))}")


print(f"The number of unique noise classes: {metadata2['noise_classes'].nunique()}")
print(f"The count of each k value: {metadata2['k'].value_counts()}")
print(f"The number of unique percussion files: {metadata2['percussion_file'].nunique()}")
print(f"The number of unique mix files: {metadata2['mix_file'].nunique()}")
print(f"The number of unique noise files: {metadata2['noise_files'].nunique()}")
print(f"The count of each noise file: {metadata2['noise_files'].value_counts()}")
print(metadata2.groupby("noise_files").size().describe())

# see the different noise files saved
noise_files = metadata2['noise_files'].str.split(',')
noise_files = [item for sublist in noise_files for item in sublist]
print(f"The number of unique noise files: {len(set(noise_files))}")

