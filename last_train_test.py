import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio
from data.utils import save_checkpoint
# Define the train function


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device="cuda", checkpoint_dir="checkpoint"):
    # Initialize metrics
    sdr_metric = SignalDistortionRatio().to(device)
    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)

    # Lists to store losses and metrics
    train_losses = []
    val_losses = []
    val_sdr = []
    val_si_sdr = []

    # Initialize learning rate scheduler (Reduce LR on Plateau based on validation loss)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose='True')

    # Early stopping criteria
    best_val_loss = float('inf')
    patience = 5

    for epoch in range(num_epochs):
        # Training step
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {
                         epoch + 1}/{num_epochs} Training", colour='green')

        for i, batch in enumerate(train_bar):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Move data to device
            mixture = batch['mixture_audio'].to(device)
            true_percussion = batch['percussion_audio'].to(device)

            # Forward pass
            output_waveform = model(mixture)['waveform']

            # Loss calculation
            loss = criterion(output_waveform, true_percussion)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()
            train_bar.set_description(
                f"Epoch {epoch + 1}/{num_epochs} Training Loss: {train_loss/(i+1):.8f}")

        # Calculate average training loss for the epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0
        val_sdr_epoch = 0
        val_si_sdr_epoch = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {
                       epoch + 1}/{num_epochs} Validation", colour='red')
        with torch.no_grad():
            for i, batch in enumerate(val_bar):
                mixture = batch['mixture_audio'].to(device)
                true_percussion = batch['percussion_audio'].to(device)

                # Forward pass
                output_waveform = model(mixture)['waveform']

                # Loss calculation
                loss = criterion(output_waveform, true_percussion)
                val_loss += loss.item()

                # Calculate SDR and SI-SDR using torchmetrics
                si_sdr = si_sdr_metric(output_waveform, true_percussion)
                sdri = sdr_metric(true_percussion, output_waveform) - \
                    sdr_metric(true_percussion, mixture)

                val_sdr_epoch += sdri.item()
                val_si_sdr_epoch += si_sdr.item()

                # Update progress bar
                val_bar.set_description(f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss/(
                    i+1):.8f}, SI-SDR: {val_si_sdr_epoch/(i+1):.8f}, SDR Improvement: {val_sdr_epoch/(i+1):.8f}")

        # Calculate average validation loss and metrics for the epoch
        val_loss /= len(val_loader)
        val_sdr_epoch /= len(val_loader)
        val_si_sdr_epoch /= len(val_loader)

        # Store validation losses and metrics
        val_losses.append(val_loss)
        val_sdr.append(val_sdr_epoch)
        val_si_sdr.append(val_si_sdr_epoch)

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Save model checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 5
            torch.save(model.state_dict(), os.path.join(
                checkpoint_dir, 'best_model_last.pth'))
            print("Model improved. Saving the model...")
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                break

        # Save the current epoch's checkpoint
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir=checkpoint_dir,
                        filename=f'checkpoint_last_epoch_{epoch}.pth')

    return train_losses, val_losses, val_sdr, val_si_sdr

# Test function


def test_model(model, test_loader, criterion, device="cuda"):
    model.eval()
    sdr_metric = SignalDistortionRatio().to(device)
    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)

    # Initialize lists to store results
    test_loss = 0
    percussion_files_l, noise_names_l, mix_names_l, k_values_l, classes_l = [], [], [], [], []
    noise_audio_l, output_waveform_l, true_percussion_l, mixtures_l, si_sdr_tensor, sdr_tensor = [], [], [], [], [], []

    test_bar = tqdm(test_loader, desc=f"Testing", colour='red')
    with torch.no_grad():
        for i, batch in enumerate(test_bar):
            # Move data to the device
            mixture = batch['mixture_audio'].to(device)
            true_percussion = batch['percussion_audio'].to(device)

            # Forward pass through the model
            output_waveform = model(mixture)['waveform']

            # Calculate the loss
            loss = criterion(output_waveform, true_percussion)
            test_loss += loss.item()

            # Calculate SDR and SI-SDR for the current batch
            si_sdr = si_sdr_metric(output_waveform, true_percussion)
            sdri = sdr_metric(true_percussion, output_waveform) - \
                sdr_metric(true_percussion, mixture)

            # Append batch data to the lists
            percussion_files_l.extend(batch['perc_name'])
            noise_names_l.extend(batch['noise_name'])
            mix_names_l.extend(batch['mix_name'])
            k_values_l.extend(batch['k'])
            classes_l.extend(batch['noise_classes'])
            noise_audio_l.extend(batch['noise_audio'])

            output_waveform_l.append(output_waveform.cpu().numpy())
            true_percussion_l.append(true_percussion.cpu().numpy())
            mixtures_l.append(mixture.cpu().numpy())

            si_sdr_tensor.append(si_sdr.item())
            sdr_tensor.append(sdri.item())

            # Update progress bar description
            test_bar.set_description(f"Testing Loss: {test_loss/(i+1):.8f}, SI-SDR: {torch.mean(
                torch.tensor(si_sdr_tensor)):.8f}, SDR Improvement: {torch.mean(torch.tensor(sdr_tensor)):.8f}")

    # Final loss and SI-SDR
    test_loss /= len(test_loader)
    final_si_sdr = torch.mean(torch.tensor(si_sdr_tensor))
    final_sdr = torch.mean(torch.tensor(sdr_tensor))

    print(f"Testing Loss: {test_loss:.8f}, Final SI-SDR: {
          final_si_sdr:.8f}, Final SDR Improvement: {final_sdr:.8f}")

    # Convert lists to tensors or numpy arrays for further processing if needed
    output_waveform = torch.cat(output_waveform_l).cpu().numpy()
    true_percussion = torch.cat(true_percussion_l).cpu().numpy()
    mixtures = torch.cat(mixtures_l).cpu().numpy()
    noise_audio = np.array(noise_audio_l)

    percussion_files = np.array(percussion_files_l)
    noise_names = np.array(noise_names_l)
    mix_names = np.array(mix_names_l)
    k_values = np.array(k_values_l)
    classes = np.array(classes_l)
    si_sdr_tensor = torch.tensor(si_sdr_tensor).numpy()
    sdr_tensor = torch.tensor(sdr_tensor).numpy()

    # Save results
    os.makedirs('results', exist_ok=True)
    np.save('results/percussion_files.npy', percussion_files)
    np.save('results/noise_names.npy', noise_names)
    np.save('results/mix_names.npy', mix_names)
    np.save('results/k_values.npy', k_values)
    np.save('results/classes.npy', classes)
    np.save('results/noise_audio.npy', noise_audio)
    np.save('results/output_waveform.npy', output_waveform)
    np.save('results/true_percussion.npy', true_percussion)
    np.save('results/mixtures.npy', mixtures)
    np.save('results/si_sdr.npy', si_sdr_tensor)

    # Return the results

    return {
        'test_loss': test_loss,
        'final_si_sdr': final_si_sdr,
        'final_sdr': final_sdr,
        'output_waveform': output_waveform,
        'true_percussion': true_percussion,
        'mixtures': mixtures,
        'percussion_files': percussion_files,
        'noise_names': noise_names,
        'mix_names': mix_names,
        'k_values': k_values,
        'classes': classes,
        'noise_audio': noise_audio,
        'si_sdr_tensor': si_sdr_tensor,
        'sdr_tensor': sdr_tensor
    }


# Plot function for losses and metrics


def plot_losses_and_metrics(train_losses, val_losses, val_sdr, val_si_sdr):
    epochs = range(1, len(train_losses) + 1)

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot training and validation loss
    axs[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
    axs[0, 0].plot(epochs, val_losses, label='Validation Loss', color='red')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Plot SDR improvement
    axs[0, 1].plot(epochs, val_sdr, label='SDR Improvement', color='green')
    axs[0, 1].set_title('Validation SDR Improvement')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('SDR Improvement')
    axs[0, 1].legend()

    # Plot SI-SDR
    axs[1, 0].plot(epochs, val_si_sdr,
                   label='Validation SI-SDR', color='purple')
    axs[1, 0].set_title('Validation SI-SDR')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('SI-SDR')
    axs[1, 0].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
