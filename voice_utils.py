import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
import json
from collections import Counter

# model_path = 'models/voice_classification_cnn_v2.h5'
# label_path = 'models/class_labels_2.json'

IMAGE_SIZE = (128, 128)  # Cập nhật để khớp với model

def plot_spectrogram(audio_path, segment_duration=3, save_dir=None, use_mel=True):
    """Chuyển audio thành các ảnh Mel Spectrogram và lưu vào thư mục."""
    y, sr = librosa.load(audio_path, sr=None)
    segment_samples = segment_duration * sr

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, start in enumerate(range(0, len(y), segment_samples)):
        end = start + segment_samples
        y_segment = y[start:end]

        if len(y_segment) < segment_samples:
            break

        plt.figure(figsize=(10, 4))

        if use_mel:
            S = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=128, fmax=8000)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
            plt.title(f'Mel Spectrogram - Segment {i+1}')
        else:
            S = np.abs(librosa.stft(y_segment))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
            plt.title(f'Spectrogram - Segment {i+1}')

        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, f'segment_{i+1}.png')
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    ipd.display(ipd.Audio(audio_path))

def predict_speaker_from_folder(folder_path, model_path, label_path):
    """Dự đoán speaker từ folder chứa nhiều ảnh spectrogram và tính độ chính xác trung bình."""

    # Load mô hình
    model = keras.models.load_model(model_path)

    # Load nhãn
    with open(label_path, 'r') as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}  # map index -> label

    # Danh sách ảnh trong folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("❌ Không tìm thấy ảnh trong folder!")
        return None

    predictions = []
    confidences = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Load và tiền xử lý ảnh
        img = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img) / 255.0  # Chuẩn hóa
        img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension

        # Dự đoán
        preds = model.predict(img_array)
        predicted_class = np.argmax(preds)
        class_labels = index_to_class  # dùng mapping từ file json đã load
        # class_labels = list(train_generator.class_indices.keys())  # Lấy nhãn class

        # predictions.append(class_labels[predicted_class])
        predictions.append(index_to_class[predicted_class])
        confidences.append(preds[0][predicted_class])  # Lưu độ tự tin

    # Lấy speaker được dự đoán nhiều nhất
    most_common_speaker, count = Counter(predictions).most_common(1)[0]

    # Tính độ chính xác trung bình
    avg_confidence = np.mean(confidences)

    return most_common_speaker, avg_confidence

def clear_folder(folder_path):
    """Xóa toàn bộ ảnh trong thư mục sau khi test xong."""
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file_path.endswith(('.png', '.jpg', '.jpeg')):
            os.remove(file_path)


# audio_path = "Data/Data_test/Voice_10/BTV Việt Hà - Dự Báo Thời Tiết.mp3"
# mel_save_dir = "Data/Temp"


# # Bước 1: Tạo Mel Spectrogram từ audio
# plot_spectrogram(audio_path, save_dir=mel_save_dir, use_mel=True)

# # Bước 2: Dự đoán speaker
# speaker, confidence = predict_speaker_from_folder(mel_save_dir, model_path, label_path)
# print(f"🔊 Dự đoán người nói: {speaker} với độ tin cậy: {confidence:.2f}")

# # Bước 3: Xoá ảnh sau khi test
# clear_folder(mel_save_dir)