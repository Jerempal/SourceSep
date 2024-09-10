# %%
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from data.utils import create_mixture, get_stft, pad_audio_center
from config import DATASET_NOISE_PATH, DATASET_PERCUSSION_PATH, DATASET_MIX_AUDIO_PATH

# Load noise metadata
metadata_noise = os.path.join(DATASET_NOISE_PATH, "UrbanSound8k.csv")
metadata_noise = pd.read_csv(metadata_noise)

# Change classID 8 to 6
metadata_noise['classID'] = metadata_noise['classID'].replace(8, 6)
# Sort by class ID and fold and reset index
metadata_noise = metadata_noise.sort_values(
    by=['classID', 'fold']).reset_index(drop=True)

# Get unique noise classes and their IDs
noise_class = metadata_noise['class'].unique()
noise_classID = metadata_noise['classID'].unique()

# Load percussion metadata
metadata_perc = os.path.join(DATASET_PERCUSSION_PATH, "metadata.csv")
metadata_perc = pd.read_csv(metadata_perc)

# Create a DataFrame for noise classes
noise_data = {
    'class': noise_class,
    'classID': noise_classID,
    'count': [len(metadata_noise[metadata_noise['class'] == c]) for c in noise_class],
}
df = pd.DataFrame(noise_data)

# Total number of percussion files
percussion_count = len(metadata_perc)

# Calculate total number of noise files
total_noise_files = df['count'].sum()

# Calculate proportions for each noise class
df['proportion'] = df['count'] / total_noise_files

# Number of noise files to select for each percussion file
noise_files_per_percussion = 50

# Create directories if they do not exist
os.makedirs(DATASET_MIX_AUDIO_PATH, exist_ok=True)

# we will save the informations of all the noised used for each percussion file
# save the informations in a csv file : percussion_file, noise_file, noise_class, mixture_file


# %%

# Create the folder if it does not exist
if not os.path.exists(DATASET_MIX_STFT_PATH):
    os.makedirs(DATASET_MIX_STFT_PATH)

# Create the folder if it does not exist
if not os.path.exists(DATASET_PERCUSSION_STFT_PATH):
    os.makedirs(DATASET_PERCUSSION_STFT_PATH)

# Create the folder if it does not exist
if not os.path.exists(DATASET_NOISE_STFT_PATH):
    os.makedirs(DATASET_NOISE_STFT_PATH)

# Iterate over each percussion file
for index, row in metadata_perc.iterrows():
    percussion_filename = row['name']
    percussion_file = os.path.join(
        DATASET_PERCUSSION_PATH, percussion_filename)

    # Create a list to store noise information
    sound_info = []

    # Iterate over each noise class
    for _, noise_row in df.iterrows():
        noise_class = noise_row['class']
        noise_count = noise_row['count']
        noise_proportion = noise_row['proportion']

        # Calculate number of noise files to select for this class
        noise_files_to_select = int(
            noise_files_per_percussion * noise_proportion)

        # Randomly select noise files from the current class
        selected_fold = np.random.randint(1, 11)
        noise_files_in_fold = metadata_noise[(metadata_noise['class'] == noise_class) &
                                             (metadata_noise['fold'] == selected_fold)]['slice_file_name'].tolist()

        if len(noise_files_in_fold) > 0:
            selected_noise_files = np.random.choice(
                noise_files_in_fold, noise_files_to_select, replace=False)

            for noise_file_name in selected_noise_files:
                noise_file = os.path.join(DATASET_NOISE_PATH, f"fold{
                                          selected_fold}", noise_file_name)

                # # Load audio files
                # percussion_audio = pad_audio_center(percussion_file)
                # noise_audio = pad_audio_center(noise_file)

                # # Calculate stft
                # stft_p = get_stft(percussion_audio)
                # stft_n = get_stft(noise_audio)

                # # create mixture
                # mixture_audio, stft_mix = create_mixture(
                #     percussion_audio, noise_audio)

                # Save sound information
                sound_info.append({
                    'percussion_file': percussion_filename,
                    'noise_file': noise_file_name,
                    # mixture file name
                    'mix_file': f"{percussion_filename}_{noise_file_name}",
                    'noise_class': noise_class,
                    'fold': selected_fold,
                })

                # Save mixture audio
                # mixture_audio_path = os.path.join(
                #     DATASET_MIX_AUDIO_PATH, f"{percussion_filename}_{noise_file_name}.wav")
                # sf.write(mixture_audio_path, mixture_audio, 7812)

                # # Save mixture stft
                # mix_stft_path = os.path.join(
                #     DATASET_MIX_STFT_PATH, f"{percussion_filename}_{noise_file_name}.npy")
                # np.save(mix_stft_path, stft_mix)

                # # Save noise stft
                # noise_stft_path = os.path.join(
                #     DATASET_NOISE_STFT_PATH, f"{noise_file_name}.npy")
                # np.save(noise_stft_path, stft_n)

                # # Save percussion stft
                # percussion_stft_path = os.path.join(
                #     DATASET_PERCUSSION_STFT_PATH, f"{percussion_filename}.npy")
                # np.save(percussion_stft_path, stft_p)

    # Save sound information to a csv file
    sound_info = pd.DataFrame(sound_info)
    sound_info.to_csv(os.path.join(
        DATASET_MIX_AUDIO_PATH, f"{percussion_filename}_info.csv"), index=False)


# We have the 387 metadata files (same number of percussion)

# we should save the metadata into a single metadata file
# we can use this metadata file to load the data into the dataset class
# we can also use it to split the data into training, validation and test sets
# we can also use it to load only the noise files of a specific class

metadata_info = []

for index, row in metadata_perc.iterrows():
    percussion_filename = row['name']
    sound_info = pd.read_csv(os.path.join(
        DATASET_MIX_AUDIO_PATH, f"{percussion_filename}_info.csv"))
    metadata_info.append(sound_info)

metadata_info = pd.concat(metadata_info)
metadata_info.to_csv(os.path.join(
    DATASET_MIX_AUDIO_PATH, "metadata.csv"), index=False)

# %%


# Shuffle the noise files to ensure randomness
metadata_noise = metadata_noise.sample(
    frac=1, random_state=42).reset_index(drop=True)

# Create a list to keep track of the usage count for each noise file
noise_usage = {row['slice_file_name']: 0 for _,
               row in metadata_noise.iterrows()}

# Create directories if they do not exist
os.makedirs(DATASET_MIX_AUDIO_PATH, exist_ok=True)

# Store metadata information
metadata_info = []

# Round-robin approach to ensure all noise files are used
noise_files = metadata_noise['slice_file_name'].tolist()
noise_files_iter = iter(noise_files)


def get_next_noise_file():
    global noise_files_iter  # Move global declaration here
    try:
        return next(noise_files_iter)
    except StopIteration:
        # Restart the iterator if we've gone through all files
        noise_files_iter = iter(noise_files)
        return next(noise_files_iter)


# Iterate over each percussion file
for index, row in metadata_perc.iterrows():
    percussion_filename = row['name']
    percussion_file = os.path.join(
        DATASET_PERCUSSION_PATH, percussion_filename)

    # Create a list to store noise information
    sound_info = []

    for _ in range(noise_files_per_percussion):  # Adjust this number as needed
        noise_file_name = get_next_noise_file()
        noise_file_row = metadata_noise[metadata_noise['slice_file_name']
                                        == noise_file_name].iloc[0]
        selected_fold = noise_file_row['fold']
        noise_file = os.path.join(DATASET_NOISE_PATH, f"fold{
                                  selected_fold}", noise_file_name)
        noise_class = noise_file_row['class']

        # Update noise usage count
        noise_usage[noise_file_name] += 1

        # Save sound information
        sound_info.append({
            'percussion_file': percussion_filename,
            'noise_file': noise_file_name,
            'mix_file': f"{percussion_filename}_{noise_file_name}",
            'noise_class': noise_class,
            'fold': selected_fold,
        })

    # Save sound information to a DataFrame and append to metadata_info
    sound_info_df = pd.DataFrame(sound_info)
    metadata_info.append(sound_info_df)

    # Save sound information to a CSV file
    sound_info_df.to_csv(os.path.join(DATASET_MIX_AUDIO_PATH, f"{
                         percussion_filename}_info.csv"), index=False)

# Concatenate all metadata info into a single DataFrame and save to a CSV file
metadata_info = pd.concat(metadata_info)
metadata_info.to_csv(os.path.join(
    DATASET_MIX_AUDIO_PATH, "metadata.csv"), index=False)

# %%
# Group by noise class to check the distribution of unique noise files per class
unique_noise_files_per_class = metadata_info.groupby('noise_class')[
    'noise_file'].nunique()
print("Unique noise files per class:")
print(unique_noise_files_per_class)

# Check the value counts of noise classes
noise_class_counts = metadata_info['noise_class'].value_counts()
print("\nNoise class counts:")
print(noise_class_counts)

# Check the fold distribution
fold_counts = metadata_info['fold'].value_counts()
print("\nFold distribution:")
print(fold_counts)

# Check the noise file usage distribution
noise_file_usage = metadata_info['noise_file'].value_counts()
print("\nNoise file usage counts:")
print(noise_file_usage)

# Check the unique counts of how many times noise files are used
print("\nUnique counts of noise file usage:")
print(noise_file_usage.unique())

# Descriptive statistics for noise file usage
print("\nDescriptive statistics of noise file usage:")
print(noise_file_usage.describe())


# %%

A = metadata_info['noise_file'].to_list()
B = metadata_noise['slice_file_name'].to_list()

# Check if all noise files are used
print(set(B) - set(A))  # should return : set()


# %%
# load metadata
metadata_original = pd.read_csv(os.path.join(
    DATASET_MIX_AUDIO_PATH, "metadata_original.csv"))


# check the distribution of unique noise files per class
unique_noise_files_per_class = metadata_original.groupby('noise_class')[
    'noise_file'].nunique()
print(unique_noise_files_per_class)

# Check the value counts of noise classes
noise_class_counts = metadata_original['noise_class'].value_counts()
print(noise_class_counts)

# stats and data visualization
print(metadata_original['fold'].value_counts())

# see how many times a noise files is used
print(metadata_original['noise_file'].value_counts())

print(metadata_original['noise_file'].value_counts().unique())
print(metadata_original['noise_file'].value_counts().describe())
# noise files used per class
print(metadata_original.groupby('noise_class')['noise_file'].value_counts())
print(metadata_original.groupby('noise_class')
      ['noise_file'].value_counts().unique())

# %%
for noise_name in noise_class:

    DATASET_PREDICTED_AUDIO_PATH = "C:\\Users\\jejep\\Desktop\\STAGE\\data\\predicted_audio_{}".format(
        noise_name)

    if not os.path.exists(DATASET_PREDICTED_AUDIO_PATH):
        os.makedirs(DATASET_PREDICTED_AUDIO_PATH)
