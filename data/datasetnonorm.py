
# class AudioMixtureDatasetWithMetadata(Dataset):
#     def __init__(self, metadata_file, k=None, noise_class=None, sample_rate=7812, n_fft=256, hop_length=64, target_loudness=-10):
#         self.metadata = metadata_file
#         self.k = k
#         self.noise_class = noise_class
#         self.noise_class_list = [
#             'air_conditioner',
#             'car_horn',
#             'children_playing',
#             'dog_bark',
#             'drilling',
#             'engine_idling',
#             'siren',
#             'jackhammer']
#         self.sample_rate = sample_rate
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.target_loudness = target_loudness

#     def __len__(self):
#         if self.noise_class is not None:
#             return len(self.metadata[self.metadata['noise_class'] == self.noise_class])
#         else:
#             return len(self.metadata)

#     def __getitem__(self, idx):
#         if self.noise_class is not None:
#             row = self.metadata[self.metadata['noise_class']
#                                 == self.noise_class].iloc[idx]
#         else:
#             row = self.metadata.iloc[idx]

#         k = self.k if self.k is not None else np.random.choice(
#             [0.5, 0.6, 0.7, 0.8, 0.9])

#         percussion_path = os.path.join(
#             DATASET_PERCUSSION_PATH, row['percussion_file'])
#         percussion_audio = pad_audio_center(percussion_path)

#         noise_path = os.path.join(DATASET_NOISE_PATH, f"fold{
#                                   row['fold']}", row['noise_file'])
#         noise_audio = pad_audio_center(noise_path)

#         mixture_audio, _ = create_mixture(percussion_audio, noise_audio, k)

#         percussion_audio /= np.max(np.abs(percussion_audio))
#         percussion_audio = k * percussion_audio

#         noise_audio /= np.max(np.abs(noise_audio))
#         noise_audio = (1 - k) * noise_audio

#         return {
#             'percussion_audio': percussion_audio,
#             'noise_audio': noise_audio,
#             'mixture_audio': mixture_audio,
#             'noise_class': torch.tensor(self.noise_class_list.index(row['noise_class'])),
#         }
