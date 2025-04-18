import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import os

def plot_spectrogram(audio_path, segment_duration=3, save_dir=None, use_mel=True):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    segment_samples = segment_duration * sr

    # Ensure save directory exists
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, start in enumerate(range(0, len(y), segment_samples)):
        end = start + segment_samples
        y_segment = y[start:end]

        if len(y_segment) < segment_samples:
            break  # Skip short segments

        plt.figure(figsize=(10, 4))

        if use_mel:
            # Compute Mel spectrogram
            S = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=128, fmax=8000)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
            plt.title(f'Mel Spectrogram - Segment {i+1}')
        else:
            # Compute regular spectrogram
            S = np.abs(librosa.stft(y_segment))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
            plt.title(f'Spectrogram - Segment {i+1}')

        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.tight_layout()

        # Save image if save_dir is provided
        if save_dir:
            save_path = os.path.join(save_dir, f'segment_{i+1}.png')
            plt.savefig(save_path)
            plt.close()  # Close figure to save memory
        else:
            plt.show()

    # Play original audio
    # ipd.display(ipd.Audio(audio_path))
    print("ĐÔN")

plot_spectrogram("Data/Data_preprocessing/voice_unknown_denoising.mp3",
                 save_dir="Data/Mel_Spectrograms/Voice_Unknown", use_mel=True)