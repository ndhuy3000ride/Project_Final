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
import soundfile as sf

def plot_spectrogram(audio_path, segment_duration=3, save_dir=None, use_mel=True):
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

def predict_speaker_from_folder(folder_path, model_path, label_path, image_size, entropy_threshold=0.4):
    model = keras.models.load_model(model_path)

    with open(label_path, 'r') as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("❌ Không tìm thấy ảnh trong folder!")
        return None

    # predictions = []
    # confidences = []
    all_predictions = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = load_img(image_path, target_size=image_size)
        img_array = img_to_array(img)
        img_array = img_array / 225.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]
        all_predictions.append(preds)

        # preds = model.predict(img_array)
        # predicted_class = np.argmax(preds)
        # predictions.append(index_to_class[predicted_class])
        # confidences.append(preds[0][predicted_class])

    # most_common_speaker, count = Counter(predictions).most_common(1)[0]
    # avg_confidence = np.mean(confidences)
    avg_preds = np.mean(all_predictions, axis=0)
    entropy = -np.sum(avg_preds * np.log2(avg_preds + 1e-10))
    max_entropy = -np.log2(1/len(avg_preds)) 
    normalized_entropy = entropy / max_entropy

    predicted_class = np.argmax(avg_preds)
    closest_speaker = index_to_class[predicted_class]
    confidence = avg_preds[predicted_class]

    if normalized_entropy > entropy_threshold:
        return "Unknown Speaker", confidence, closest_speaker
    else:
        return closest_speaker, confidence, None

def clear_folder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file_path.endswith(('.png', '.jpg', '.jpeg')):
            os.remove(file_path)

def remove_silence_and_save(audio_path, save_path, top_db=30):
    y, sr = librosa.load(audio_path, sr=None)
    intervals = librosa.effects.split(y, top_db=top_db)
    y_speech = np.concatenate([y[start:end] for start, end in intervals])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sf.write(save_path, y_speech, sr)
