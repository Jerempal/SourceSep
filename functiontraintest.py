import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio

# Assuming your model and dataset classes are already defined
# from your_model_file import ResUNetv2, PreComputedMixtureDataset

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, device="cuda", patience=5):
    train_losses = []
    val_losses = []
    si_sdr_values = []
    sdri_values = []
    best_val_loss = np.inf
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training Loss: {train_loss:.8f}", colour='green')
        for i, batch in enumerate(train_bar):
            optimizer.zero_grad()

            # Move data to device
            mixture = batch['mixture_audio'].to(device)
            true_percussion = batch['percussion_audio'].to(device)

            # Forward pass
            output_waveform = model(mixture)['waveform']
            loss = criterion(output_waveform, true_percussion)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_description(f"Epoch {epoch + 1}/{num_epochs} Training Loss: {train_loss/(i+1):.8f}")

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        epoch_si_sdr = 0
        epoch_sdri = 0
        
        sdr_metric = SignalDistortionRatio().to(device)
        si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss:.8f}", colour='red')
            for i, batch in enumerate(val_bar):
                mixture = batch['mixture_audio'].to(device)
                true_percussion = batch['percussion_audio'].to(device)

                output_waveform = model(mixture)['waveform']
                loss = criterion(output_waveform, true_percussion)
                val_loss += loss.item()

                # Calculate SI-SDR and SDRi 
                si_sdr = si_sdr_metric(output_waveform, true_percussion)
                sdri = sdr_metric(true_percussion, output_waveform) - sdr_metric(true_percussion, mixture)
                
                # Accumulate SI-SDR and SDRi for averaging
                epoch_si_sdr += si_sdr.item()
                epoch_sdri += sdri.item()
                
                val_bar.set_description(f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss/(i+1):.8f}, SI-SDR: {si_sdr:.8f}, SDRi: {sdri:.8f}")
                
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Average SI-SDR and SDRi for the epoch
            avg_si_sdr = epoch_si_sdr / len(val_loader)
            avg_sdri = epoch_sdri / len(val_loader)
            si_sdr_values.append(avg_si_sdr)
            sdri_values.append(avg_sdri)

        scheduler.step(val_loss)

        # Save checkpoint if the model improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 5
            torch.save(model.state_dict(), 'best_model_last.pth')
            print("Model improved. Saving the model...")
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                break

    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
    
    # Plot the SI-SDR and SDR Improvement
    plt.figure(figsize=(10, 5))
    plt.plot(si_sdr_values, label='SI-SDR')
    plt.plot(sdri_values, label='SDR Improvement')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('SI-SDR and SDR Improvement Over Epochs')
    plt.show()

    
def test_model(model, test_loader, criterion, device="cuda"):
    model.eval()
    test_loss = 0
    percussion_files_l, noise_names_l, mix_names_l, k_values_l, classes_l = [], [], [], [], []
    noise_audio_l, output_waveform_l, true_percussion_l, mixtures_l, si_sdr_tensor = [], [], [], [], []

    sdr_metric = SignalDistortionRatio().to(device)
    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)

    test_bar = tqdm(test_loader, desc=f"Testing Loss: {test_loss:.8f}", colour='red')
    with torch.no_grad():
        for i, batch in enumerate(test_bar):
            mixture = batch['mixture_audio'].to(device)
            true_percussion = batch['percussion_audio'].to(device)

            # Forward pass
            output_waveform = model(mixture)['waveform']
            loss = criterion(output_waveform, true_percussion)
            test_loss += loss.item()

            # Calculate SI-SDR
            si_sdr = si_sdr_metric(output_waveform, true_percussion)
            si_sdr_tensor.append(si_sdr.mean().item())

            # Save relevant batch data for later analysis
            percussion_files_l.extend(batch['perc_name'])
            noise_names_l.extend(batch['noise_name'])
            mix_names_l.extend(batch['mix_name'])
            k_values_l.extend(batch['k'])
            classes_l.extend(batch['noise_classes'])
            noise_audio_l.extend(batch['noise_audio'])

            output_waveform_l.append(output_waveform.cpu().numpy())
            true_percussion_l.append(true_percussion.cpu().numpy())
            mixtures_l.append(mixture.cpu().numpy())

            test_bar.set_description(f"Testing Loss: {test_loss/(i+1):.8f}, SI-SDR: {np.mean(si_sdr_tensor):.8f}")

    test_loss /= len(test_loader)
    final_si_sdr = np.mean(si_sdr_tensor)
    print(f"Testing Loss: {test_loss:.8f}, SI-SDR: {final_si_sdr:.8f}")

    # Convert lists to numpy arrays for saving
    percussion_files = np.array(percussion_files_l)
    noise_names = np.array(noise_names_l)
    mix_names = np.array(mix_names_l)
    k_values = np.array(k_values_l)
    classes = np.array(classes_l)

    noise_audio = np.array(noise_audio_l)
    output_waveform = np.concatenate(output_waveform_l, axis=0)
    true_percussion = np.concatenate(true_percussion_l, axis=0)
    mixtures = np.concatenate(mixtures_l, axis=0)
    si_sdr_tensor = np.array(si_sdr_tensor)

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

# Example usage:

# Initialize model, optimizer, criterion, scheduler
model = ResUNetv2(in_c=1, out_c=32).to("cuda")
optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True, fused=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
criterion = torch.nn.L1Loss()

# Assuming your DataLoader instances (train_loader, val_loader, test_loader) are defined

# Train and validate
train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, device="cuda", patience=5)

# Load best model and test
model.load_state_dict(torch.load('best_model_last.pth'))
test_model(model, test_loader, criterion, device="cuda")
