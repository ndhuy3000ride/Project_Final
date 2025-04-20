import streamlit as st
import os
import tempfile
from voice_utils import plot_spectrogram, predict_speaker_from_folder, clear_folder, remove_silence_and_save

MODEL_PATH = 'models/voice_classification_cnn_v3.h5'
LABEL_PATH = 'models/class_labels_2.json'
TEMP_IMAGE_DIR = 'Data/Temp'

st.set_page_config(page_title="Voice Classifier", page_icon="🔊")

st.title("🔊 Voice Speaker Classification")
st.write("Upload a voice recording (.wav or .mp3) to identify the speaker.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[1]

# Lưu file tạm với đúng đuôi
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        temp_audio_path = tmp.name

# Tạo đường dẫn file sau khi xử lý im lặng (luôn lưu thành .wav)
    denoised_audio_path = temp_audio_path.replace(ext, "_clean.wav")

# Phát lại file đã xử lý im lặng, hoặc dùng temp_audio_path nếu muốn nghe bản gốc
    st.audio(temp_audio_path)
    with st.spinner("Processing audio..."):
        # 1. Khử khoảng lặng
        remove_silence_and_save(temp_audio_path, denoised_audio_path)

        # 2. Tạo ảnh Mel từ file đã xử lý
        plot_spectrogram(denoised_audio_path, save_dir=TEMP_IMAGE_DIR)

        # 3. Dự đoán
        speaker, confidence = predict_speaker_from_folder(TEMP_IMAGE_DIR, MODEL_PATH, LABEL_PATH)

        # 4. Dọn dẹp
        clear_folder(TEMP_IMAGE_DIR)

    if speaker:
        st.success(f"✅ Predicted Speaker: **{speaker}**")
        st.info(f"Confidence: **{confidence:.2f}**")
    else:
        st.error("❌ No spectrogram images found for prediction.")
