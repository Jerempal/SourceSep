import time
import torch
from tqdm import tqdm
from metrics_loss import spectral_loss, SDR_i, SISDR
from data.utils import audio_from_spectrogram


def train(model, train_loader, val_loader, num_epochs, optimizer, criterion, device="cuda"):
    # Move model to device
    model.to(device)

    # Initialize the best validation loss
    best_val_loss = float('inf')

    # Initialize lists to store losses and accuracies
    train_losses = []
    val_losses = []

    # Train the model
    for epoch in range(num_epochs):
        start_time = time.time()

        # Set model to training mode
        model.train()

        # Initialize variables to store loss and accuracy
        train_loss = 0.0

        # Initialize tqdm progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {
                         epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # Iterate over the training data
        for i, data in enumerate(train_bar):

            if criterion == spectral_loss:
                features = data['mixture_stft'].to(device)
                percussion_stft = data['percussion_stft'].to(device)
                mask = model(features)

                # Calculate the output (prediction)
                predicted_mag = mask[:, 0, :, :] * torch.abs(features)
                predicted_phase = mask[:, 1, :, :] * torch.angle(features)

                # Convert the output to complex
                predicted_stft = predicted_mag * \
                    torch.exp(1j * predicted_phase)

                # Calculate the loss
                loss = criterion(predicted_stft, percussion_stft)

            else:
                # mixture_audio = data['mixture_audio'].to(device)
                # percussion_audio = data['percussion_audio'].to(device)

                mixture_audio = data['mixtures'].to(device)

                # Calculate the output (prediction)
                predicted_audio = model(mixture_audio)
                percussion_pred = predicted_audio['waveform'][:, 1, :]
                # Calculate the loss
                # loss = criterion(predicted_audio, percussion_audio)
                loss = criterion(
                    percussion_pred, mixture_audio[:, 1, :])

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            # Update the training loss
            train_loss += loss.item()

            # Update the tqdm progress bar description
            train_bar.set_description(
                f"Epoch {epoch + 1}/{num_epochs}, Average training Loss: {train_loss / (i + 1):.4f}")

        # Calculate the average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Set model to evaluation mode
        model.eval()

        # Initialize variables to store loss and accuracy
        val_loss = 0.0

        SDRi_list = []

        SISDR_list = []

        # Disable gradient computation
        with torch.no_grad():
            # Initialize tqdm progress bar
            val_bar = tqdm(val_loader, desc=f"Epoch {
                           epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

            # Iterate over the validation data
            for i, data in enumerate(val_bar):
                mixture_audio = data['mixtures'].to(device)

                # Calculate the output (prediction)
                predicted_audio = model(mixture_audio)

                percussion_pred = predicted_audio['waveform'][:, 1, :]
                mixture_true = mixture_audio[:, 0, :]
                percussion_true = mixture_audio[:, 1, :]
                predicted_audio = percussion_pred

                # Calculate the loss
                loss = criterion(
                    percussion_pred, mixture_audio[:, 1, :])

                # Update the validation loss
                val_loss += loss.item()

                # Calculate the signal-to-distortion ratio improvement
                sdr_i = SDR_i(
                    percussion_true, predicted_audio, mixture_true)
                SDRi_list.append(sdr_i)

                # Calculate the scale-invariant source-to-distortion ratio
                sisdr = SISDR(
                    percussion_true, predicted_audio)
                SISDR_list.append(sisdr)

                # Update the tqdm progress bar description
                val_bar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs}, Average validation Loss: {val_loss / (i + 1):.4f}")

        # Calculate the average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Save the model with the best validation loss (early stopping)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.{}_with_val_loss{}.pth".format(
                epoch, val_loss))

        # Print the epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {
              train_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {time.time() - start_time:.2f}s")

    return train_losses, val_losses, SDRi_list, SISDR_list
