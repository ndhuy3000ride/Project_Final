import os
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def remove_silence(audio_path, top_db=30):
    """
    Loại bỏ khoảng im lặng trong file âm thanh.
    Trả về tín hiệu âm thanh đã được xử lý và sample rate.
    """
    y, sr = librosa.load(audio_path, sr=None)
    intervals = librosa.effects.split(y, top_db=top_db)
    y_speech = np.concatenate([y[start:end] for start, end in intervals])
    return y_speech, sr


def save_audio(y, sr, save_path):
    """
    Lưu âm thanh xuống file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sf.write(save_path, y, sr)
    print(f"✅ Đã lưu file âm thanh sau khi loại khoảng im lặng: {save_path}")


def generate_mel_spectrogram(y, sr, segment_duration, save_dir, prefix="segment"):
    """
    Tạo và lưu ảnh Mel Spectrogram từ tín hiệu âm thanh.
    """
    segment_samples = segment_duration * sr
    os.makedirs(save_dir, exist_ok=True)

    for i, start in enumerate(range(0, len(y), segment_samples)):
        end = start + segment_samples
        y_segment = y[start:end]

        if len(y_segment) < segment_samples:
            break  # Bỏ qua các đoạn ngắn

        # Tính Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Vẽ và lưu ảnh
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
        plt.title(f'Mel Spectrogram - {prefix}_{i+1}')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{prefix}_{i+1}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"🖼️ Đã lưu ảnh: {save_path}")


def process_audio_file(audio_path, output_audio_path, output_image_dir, segment_duration=3, top_db=30):
    """
    Xử lý một file âm thanh: loại bỏ khoảng lặng, lưu file mới và tạo ảnh Mel Spectrogram.
    """
    y, sr = remove_silence(audio_path, top_db=top_db)
    save_audio(y, sr, output_audio_path)
    generate_mel_spectrogram(y, sr, segment_duration, output_image_dir, prefix=os.path.splitext(os.path.basename(audio_path))[0])


def process_directory(input_dir, output_audio_dir, output_image_dir, segment_duration=3, top_db=30, exts=('.wav', '.mp3')):
    """
    Xử lý toàn bộ file trong thư mục đầu vào.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(exts):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                
                output_audio_path = os.path.join(output_audio_dir, relative_path, file)
                output_image_subdir = os.path.join(output_image_dir, relative_path, os.path.splitext(file)[0])

                print(f"\n📁 Đang xử lý: {input_path}")
                process_audio_file(input_path, output_audio_path, output_image_subdir, segment_duration, top_db)
