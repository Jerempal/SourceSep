

# %%
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss

# on utilise precision_recall_fscore_support pour calculer les scores de précision, recall et f1-score (multi-label)
# accuracy_score pour calculer l'accuracy (multi-label)
# hamming_loss pour calculer la hamming loss (multi-label)

# on utilise classification_report pour afficher les résultats
# on utilise multilabel_confusion_matrix pour afficher les matrices

# Test the model
model.eval()
test_loss = 0

# confusion matrix for multilabel classification
# correct = 0
# total = 0
# all_preds = []
# all_labels = []

test_bar = tqdm(test_loader, desc=f"Testing Loss: {
                test_loss:.4f}", colour='red')
with torch.no_grad():
    for i, batch in enumerate(test_bar):
        # Move data to device
        mixture = batch['mixture_audio'].to(device)
        # mix_stft = batch['mix_stft'].to(device)
        true_percussion = batch['percussion_audio'].to(device)
        perc_names = batch['perc_name']
        
        # true_percussion_stft = batch['perc_stft'].to(device)
        # true_class = batch['noise_labels'].to(device)

        # Forward pass
        # output, class_output = model(torch.abs(mix_stft))
        output_waveform = model(mixture)
        output_waveform = output_waveform['waveform']

        # Reconstruct the complex spectrogram
        # sep_output = SpectrogramReconstructor().reconstruct(
        #     output['mag_mask'], output['real_mask'], output['imag_mask'], mix_stft)
        # # percussion_sep = istft(sep_output, n_fft=256, hop_length=64)

        # Calculate the loss
        # loss = criterion(percussion_sep, class_output, true_percussion, true_class)
        # loss = criterion(sep_output, class_output, true_percussion_stft,
        #                  true_class, alpha=0.7, beta=0.3, spectrogram_loss=True)

        loss = criterion(output_waveform, true_percussion)
        test_loss += loss.item()

        # Calculate multi-label classification accuracy
        # predicted = (torch.sigmoid(class_output) > 0.5).float()
        # # Total for a multi-label classification:
        # total += true_class.size(0) * true_class.size(1)
        # correct += (predicted == true_class).float().sum().item()

        # all_preds.extend(predicted.cpu().numpy())
        # all_labels.extend(true_class.cpu().numpy())

        test_bar.set_description(
            f"Testing Loss: {test_loss/(i+1):.4f}")

    test_loss /= len(test_loader)
    # accuracy = correct / total

    # concatenate the names of the percussions
    perc_names = [item for sublist in perc_names for item in sublist]

    perc_names = np.array(perc_names)

# %%
# plot the true and predicted waveforms

for i in range(30):
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.plot(mixture[i, 0].cpu().numpy(), label='Mix', color='black')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(true_percussion[i, 0].cpu().numpy(),
             label='True Percussion', color='red')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(output_waveform[i, 0].cpu().numpy(),
             label='Predicted Percussion', color='blue')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.show()


# %%
# plot spectrograms

for i in range(30):
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(
        np.abs(librosa.stft(mixture[i, 0].cpu().numpy(), n_fft=256, hop_length=64)), ref=np.max), y_axis='linear', x_axis='time', sr=7812)

    plt.colorbar(format='%+2.0f dB')
    plt.title('Mix')
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(
        np.abs(librosa.stft(true_percussion[i, 0].cpu().numpy(), n_fft=256, hop_length=64)), ref=np.max), y_axis='linear', x_axis='time', sr=7812)

    plt.colorbar(format='%+2.0f dB')
    plt.title('True Percussion')
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(
        np.abs(librosa.stft(output_waveform[i, 0].cpu().numpy(), n_fft=256, hop_length=64)), ref=np.max), y_axis='linear', x_axis='time', sr=7812)

    plt.colorbar(format='%+2.0f dB')
    plt.title('Predicted Percussion')
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    plt.tight_layout
    plt.show()


# %%

