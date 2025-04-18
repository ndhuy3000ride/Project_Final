import os
import librosa
import soundfile as sf
import numpy as np

def remove_silence_and_save(audio_path, save_path, top_db=30):
    """
    Loại bỏ khoảng im lặng trong file âm thanh và lưu ra file mới.
    
    Args:
        audio_path (str): Đường dẫn tới file âm thanh đầu vào (.wav)
        save_path (str): Đường dẫn lưu file âm thanh đầu ra (.wav)
        top_db (int): Ngưỡng dB để xác định khoảng im lặng (mặc định 30)
    """
    # Load file
    y, sr = librosa.load(audio_path, sr=None)

    # Loại bỏ khoảng trống
    intervals = librosa.effects.split(y, top_db=top_db)
    y_speech = np.concatenate([y[start:end] for start, end in intervals])

    # Tạo thư mục nếu cần
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Lưu file âm thanh mới
    sf.write(save_path, y_speech, sr)
    print(f"✅ Đã lưu file sau khi loại khoảng im lặng: {save_path}")

remove_silence_and_save(
    audio_path='Data/Data_Raw/voice_unknown.mp3',
    save_path='Data/Data_preprocessing/voice_unknown_denoising.mp3'
)