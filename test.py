import os
import pandas as pd
import torch
import soundfile as sf
from tqdm import tqdm
from data.utils import audio_from_spectrogram
from metrics_loss import SDR_i, SISDR

noise_name = input("Enter the noise class name: ")
DATASET_PREDICTED_AUDIO_PATH = "C:\\Users\\jejep\\Desktop\\STAGE\\data\\predicted_audio_{}".format(
    noise_name)

# load_state_dict(torch.load(
#     "best_model.6_with_val_loss1.2521952390670776.pth", weights_only=True))

# Define the test function


def test(model, test_loader, criterion, device="cuda"):

    # Set the model to device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Initialize the signal-to-distortion ratio improvement list
    SDRi_list = []

    # Initialize the SI-SDR list
    SISDR_list = []

    # Initialize the mixtures names list
    mixtures_names = []

    # Disable gradient computation
    with torch.no_grad():

        # Initialize the loss
        test_loss = 0.0

        # Initialize tqdm progress bar
        test_bar = tqdm(test_loader, desc=f"Testing Loss: {test_loss:.4f}")

        # Iterate over the test data
        for i, data in enumerate(test_bar):

            mixtures = data['mixtures'].to(device)
            percussion_true = mixtures[:, 1, :]
            mixture_true = mixtures[:, 0, :]
            
            mixture_name = data['mixture_name']
            percussion_name = data['percussion_name']

            # Forward pass
            predicted_audio = model(mixtures)

            percussion_pred = predicted_audio['waveform'][:, 1, :]

            # Calculate the loss
            loss = criterion(percussion_pred, percussion_true)

            # Update the test loss
            test_loss += loss.item()

            # Calculate the signal-to-distortion ratio improvement
            # sdr_i = SDR_i(percussion_true, predicted_audio, mixture_true)
            sdr_i = SDR_i(percussion_true, percussion_pred, mixture_true)
            SDRi_list.append(sdr_i)

            # Calculate the SI-SDR
            # sisdr = SISDR(percussion_true, predicted_audio)
            sisdr = SISDR(percussion_true, percussion_pred)
            SISDR_list.append(sisdr)
            
            # save the predicted audio with the name of the true percussion audio
            for k in range(len(percussion_pred)):
                predicted_audio_path = os.path.join(
                    DATASET_PREDICTED_AUDIO_PATH, f"{percussion_name[k].split('.wav')[0]}_mix{mixture_name[k].split('.wav')[1]}_predicted.wav")
                sf.write(predicted_audio_path, percussion_pred[k].cpu(), 7812)

            # store the mixtures names
            mixtures_names.extend(mixture_name)

            # Update the progress bar description
            test_bar.set_description(
                f"Testing Loss: {test_loss / (i + 1):.4f}")

        # Calculate the average loss
        test_loss /= len(test_loader)

        return test_loss, SDRi_list, SISDR_list, mixtures_names


def process_metadata(metadata, predicted_audio_path, mixtures_names, noise_name):

    df = pd.DataFrame(mixtures_names)

    df['percussion name'] = df[0].apply(lambda x: x.split('.wav')[0])
    df.rename(columns={0: 'mix name'}, inplace=True)
    df['noise name'] = df['mix name'].apply(lambda x: x.split('.wav_')[1])
    df['predicted name'] = df['percussion name'].apply(lambda x: f"{x}_mix{
                                                       df[df['percussion name'] == x]['mix name'].values[0].split('.wav')[1]}_predicted.wav")

    for noise_file in df['noise name']:
        row = metadata[metadata['noise_file'] == noise_file]
        df.loc[df['noise name'] == noise_file, 'fold'] = row['fold'].values[0]
        df.loc[df['noise name'] == noise_file,
               'noise_class'] = row['noise_class'].values[0]

    df['fold'] = df['fold'].astype(int)

    df.to_csv(os.path.join(predicted_audio_path, "metadata_pred_{noise_class}.csv".format(
        noise_class=noise_name)), index=False)

    return df, noise_name
