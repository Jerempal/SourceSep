
class AudioMixtureDatasetWithDirs(Dataset):
    def __init__(self, percussion_dir=DATASET_PERCUSSION_PATH, noise_dir=DATASET_NOISE_PATH, sample_rate=7812, n_fft=256, hop_length=64, target_loudness=-30, max_noise_sources=3):
        self.percussion_dir = percussion_dir
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_loudness = target_loudness
        self.max_noise_sources = max_noise_sources

        self.percussion_files = [f for f in os.listdir(
            percussion_dir) if f.endswith('.wav')]

        self.noise_files = {}
        for fold in range(1, 11):
            self.noise_files[fold] = [f for f in os.listdir(
                os.path.join(noise_dir, f"fold{fold}")) if f.endswith('.wav')]

        self.noise_class_list = [
            'air_conditioner',
            'car_horn',
            'children_playing',
            'dog_bark',
            'drilling',
            'engine_idling',
            'siren',
            'jackhammer']

    def __len__(self):
        return len(self.percussion_files)

    def __getitem__(self, idx):
        percussion_path = os.path.join(
            self.percussion_dir, self.percussion_files[idx])
        percussion_audio = pad_audio_center(percussion_path)
        percussion_audio /= np.max(np.abs(percussion_audio))

        num_noise_sources = np.random.randint(1, self.max_noise_sources + 1)
        selected_noise_files = []
        selected_noise_classes = []

        for _ in range(num_noise_sources):
            fold = np.random.randint(1, 11)
            noise_file = np.random.choice(
                self.noise_files[fold], replace=False)
            selected_noise_files.append(
                {'fold': fold, 'noise_file': noise_file})

            noise_class = self.get_noise_class(
                noise_file)  # Implement this function
            selected_noise_classes.append(noise_class)

        noise = np.zeros_like(percussion_audio)
        for noise_file in selected_noise_files:
            noise_path = os.path.join(self.noise_dir, f'fold{
                                      noise_file["fold"]}', noise_file["noise_file"])
            noise_src = pad_audio_center(noise_path)
            noise_src /= np.max(np.abs(noise_src))
            noise += noise_src

        noise /= np.max(np.abs(noise))

        k = np.random.uniform(0.5, 0.9)
        scaled_perc = k * percussion_audio
        scaled_noise = (1 - k) * noise
        mixture = scaled_perc + scaled_noise
        mixture /= np.max(np.abs(mixture))

        percussion_stft = librosa.stft(
            scaled_perc, n_fft=self.n_fft, hop_length=self.hop_length)
        mixture_stft = librosa.stft(
            mixture, n_fft=self.n_fft, hop_length=self.hop_length)

        return {
            'percussion_audio': scaled_perc,
            'noise_audio': scaled_noise,
            'mixture_audio': mixture,
            'percussion_stft': percussion_stft,
            'mixture_stft': mixture_stft,
            'noise_classes': torch.tensor([self.noise_class_list.index(nc) for nc in selected_noise_classes]),
        }

    def get_noise_class(self, noise_file):
        for noise_class in self.noise_class_list:
            if noise_class in noise_file:
                return noise_class
        return 'unknown'
