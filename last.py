Y_waveform_np = Y_waveform.squeeze().detach().cpu().numpy()
mixtures = batch['mixture_audio']
noise = batch['noise_audio']

plt.figure(figsize=(10, 10))
for idx in range(Y_waveform_np.shape[0]):
    plt.subplot(4, 1, 1)
    plt.plot(Y_waveform_np[idx])
    plt.title('Generated Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 2)
    plt.plot(target_waveform[idx].cpu().numpy())
    plt.title('Target Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 3)
    plt.plot(mixtures[idx].cpu().numpy())
    plt.title('Mixture Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 4)
    plt.plot(noise[idx].cpu().numpy())
    # plt.title('Noise Audio with class: ' + str(batch['noise_class']))
    plt.title('Noise Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

# %%

target_waveform = target_waveform.cpu().numpy()
noise = noise.cpu().numpy()

# normalize the audio
for idx in range(Y_waveform_np.shape[0]):
    Y_waveform_np[idx] /= np.max(np.abs(Y_waveform_np[idx]))
    target_waveform[idx] /= np.max(np.abs(target_waveform[idx]))
    noise[idx] /= np.max(np.abs(noise[idx]))


# %%
plt.figure(figsize=(10, 10))
for idx in range(Y_waveform_np.shape[0]):
    plt.subplot(4, 1, 1)
    plt.plot(Y_waveform_np[idx])
    plt.title('Generated Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 2)
    plt.plot(target_waveform[idx])
    plt.title('Target Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 3)
    plt.plot(mixtures[idx].cpu().numpy())
    plt.title('Mixture Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 4)
    plt.plot(noise[idx])
    # plt.title('Noise Audio with class: ' + str(batch['noise_class']))
    plt.title('Noise Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
# %%

for idx in range(Y_waveform_np.shape[0]):
    sf.write(f'generated_audio_{idx}.wav', Y_waveform_np[idx], 7812)
    sf.write(f'target_audio_{idx}.wav', target_waveform[idx], 7812)
    sf.write(f'mixture_audio_{idx}.wav', mixtures[idx].cpu().numpy(), 7812)
    sf.write(f'noise_audio_{idx}.wav', noise[idx], 7812)

# %%
