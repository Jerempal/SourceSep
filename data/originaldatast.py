
class AudioMixtureDatasetWithMetadata(Dataset):
    def __init__(self, metadata_file, k=None, noise_classes=None, max_noise_classes=3, sample_rate=7812, n_fft=256, hop_length=64, target_loudness=-23):
        self.metadata = metadata_file
        self.k = k
        self.noise_classes = noise_classes
        self.max_noise_classes = max_noise_classes
        self.noise_class_list = [
            'air_conditioner',
            'car_horn',
            'children_playing',
            'dog_bark',
            'drilling',
            'engine_idling',
            'siren',
            'jackhammer']
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_loudness = target_loudness
        self.meter = pyln.Meter(sample_rate)  # Loudness meter

    def __len__(self):
        return len(self.metadata)

    def _load_and_pad_audio(self, audio_path):
        # audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        # return librosa.util.fix_length(audio, size=self.sample_rate)  # Pad or trim audio
        return pad_audio_center(audio_path)

    def _normalize_loudness(self, audio):

        # Check if the audio is too silent
        if self.meter.integrated_loudness(audio) == -np.inf:
            # Apply peak normalization
            new_audio = pyln.normalize.peak(audio, self.target_loudness)

            loudness = self.meter.integrated_loudness(new_audio)

            normalized_audio = pyln.normalize.loudness(
                new_audio, loudness, self.target_loudness)
                
        else:
            # Normal loudness normalization
            loudness = self.meter.integrated_loudness(audio)
            normalized_audio = pyln.normalize.loudness(
                audio, loudness, self.target_loudness)

        # Ensure the audio does not clip by scaling to [-1, 1] if necessary
        max_amplitude = max(abs(normalized_audio))
        if max_amplitude > 1.0:
            normalized_audio = normalized_audio / \
                max_amplitude  # Scale back to [-1, 1]
                
        return normalized_audio

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Randomly select k if not provided
        k = self.k if self.k is not None else np.random.choice(
            [0.5, 0.6, 0.7, 0.8, 0.9])

        # Load percussion audio
        percussion_path = os.path.join(
            DATASET_PERCUSSION_PATH, row['percussion_file'])
        percussion_audio = self._load_and_pad_audio(percussion_path)

        # Normalize percussion loudness
        percussion_audio = self._normalize_loudness(percussion_audio)
        percussion_audio = k * percussion_audio

        # Randomly select multiple noise classes if not provided
        # Ensure that noise_class_list is treated as a list of full noise class names
        noise_classes = self.noise_classes if self.noise_classes else np.random.choice(
            self.noise_class_list, size=np.random.randint(1, self.max_noise_classes + 1), replace=False)

        # Print the selected noise classes to confirm
        print(f"Selected noise classes: {noise_classes}")

        noise_audio_total = np.zeros_like(percussion_audio)
        noise_labels = np.zeros(len(self.noise_class_list), dtype=np.float32)

        # Ensure that you're handling full noise class names, not individual characters
        for noise_class in noise_classes:
            if noise_class not in self.noise_class_list:
                print(f"Warning: Invalid noise class '{
                      noise_class}'. Skipping.")
                continue

            # Filter the metadata for the current noise class
            noise_class_data = self.metadata[self.metadata['noise_class']
                                             == noise_class]

            # Check if the filtered data is empty
            if len(noise_class_data) == 0:
                print(f"Warning: No data found for noise class '{
                      noise_class}'. Skipping.")
                continue

            # Randomly sample one row from the filtered data
            noise_row = noise_class_data.sample(n=1).iloc[0]

            noise_path = os.path.join(DATASET_NOISE_PATH, f"fold{
                                      noise_row['fold']}", noise_row['noise_file'])
            noise_audio = self._load_and_pad_audio(noise_path)

            # Normalize noise loudness and add to total noise
            noise_audio = self._normalize_loudness(noise_audio)
            noise_audio_total += (1 - k) * noise_audio

            # clip the noise audio
            noise_audio_total /= np.max(np.abs(noise_audio_total))

            # Set the noise label to 1 for the current noise class
            noise_labels[self.noise_class_list.index(noise_class)] = 1.0

        # Create mixture
        mixture_audio = percussion_audio + noise_audio_total
        # Normalize to prevent clipping
        mixture_audio = mixture_audio / np.max(np.abs(mixture_audio))

        return {
            'percussion_audio': torch.tensor(percussion_audio, dtype=torch.float32),
            'noise_audio': torch.tensor(noise_audio_total, dtype=torch.float32),
            'mixture_audio': torch.tensor(mixture_audio, dtype=torch.float32),
            # Multi-label classification
            'noise_labels': torch.tensor(noise_labels, dtype=torch.float32),
        }

