#%%
import os
import pandas as pd
import numpy as np
import soundfile as sf


from utils import create_mixture, get_stft, pad_audio_center
from config import DATASET_NOISE_PATH, DATASET_PERCUSSION_PATH, DATASET_MIX_AUDIO_PATH, DATASET_MIX_STFT_PATH, DATASET_PERCUSSION_STFT_PATH, DATASET_NOISE_STFT_PATH

# %%
# where to find the noise
metadata_noise = os.path.join(DATASET_NOISE_PATH, "UrbanSound8k.csv")
metadata_noise = pd.read_csv(metadata_noise)

# sort the noise class by class ID
metadata_noise = metadata_noise.sort_values(by='classID')


#%%
metadata_perc = os.path.join(DATASET_PERCUSSION_PATH, "metadata.csv")
metadata_perc = pd.read_csv(metadata_perc)

# Example dataframe for noise classes
noise_data = {
    'class': ['dog_bark', 'children_playing', 'air_conditioner', 'engine_idling',
              'jackhammer', 'drilling', 'siren', 'car_horn'],
    'count': [1000, 1000, 1000, 1000, 1000, 1000, 929, 429]
}
df = pd.DataFrame(noise_data)

# Example paths to noise files (adjust paths according to your setup)
noise_folder = DATASET_NOISE_PATH  # path to the folder containing the noise files

# %%

# Total number of percussion files
percussion_count = 387

# Calculate total number of noise files
total_noise_files = df['count'].sum()

# Calculate proportions for each noise class
df['proportion'] = df['count'] / total_noise_files

# every percussion files will get 40 noise files 5 from each class
# this will give us a total number of 387 * 24 = 9288 noise files
# Number of noise files to select for each percussion file
noise_files_per_percussion = 40

# we will save the informations of all the noised used for each percussion file
# save the informations in a csv file : percussion_file, noise_file, noise_class, mixture_file


# %%

# Create the folder if it does not exist
if not os.path.exists(DATASET_MIX_AUDIO_PATH):
    os.makedirs(DATASET_MIX_AUDIO_PATH)

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

                # Load audio files
                percussion_audio = pad_audio_center(percussion_file)
                noise_audio = pad_audio_center(noise_file)

                # Calculate stft
                stft_p = get_stft(percussion_audio)
                stft_n = get_stft(noise_audio)

                # create mixture
                mixture_audio, stft_mix = create_mixture(
                    percussion_audio, noise_audio)

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
                mixture_audio_path = os.path.join(
                    DATASET_MIX_AUDIO_PATH, f"{percussion_filename}_{noise_file_name}.wav")
                sf.write(mixture_audio_path, mixture_audio, 7812)

                # Save mixture stft
                mix_stft_path = os.path.join(
                    DATASET_MIX_STFT_PATH, f"{percussion_filename}_{noise_file_name}.npy")
                np.save(mix_stft_path, stft_mix)

                # Save noise stft
                noise_stft_path = os.path.join(
                    DATASET_NOISE_STFT_PATH, f"{noise_file_name}.npy")
                np.save(noise_stft_path, stft_n)

                # Save percussion stft
                percussion_stft_path = os.path.join(
                    DATASET_PERCUSSION_STFT_PATH, f"{percussion_filename}.npy")
                np.save(percussion_stft_path, stft_p)

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

noise_class = ['dog_bark', 'children_playing', 'car_horn', 'air_conditioner',
               'siren', 'engine_idling', 'jackhammer', 'drilling']

for noise_name in noise_class:
    
    DATASET_PREDICTED_AUDIO_PATH = "C:\\Users\\jejep\\Desktop\\STAGE\\data\\predicted_audio_{}".format(
        noise_name)

    if not os.path.exists(DATASET_PREDICTED_AUDIO_PATH):
        os.makedirs(DATASET_PREDICTED_AUDIO_PATH)


# %%
