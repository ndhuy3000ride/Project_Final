# app.py

import streamlit as st
import os
import tempfile
from voice_utils import plot_spectrogram, predict_speaker_from_folder, clear_folder

MODEL_PATH = 'models/voice_classification_cnn_v2.h5'
LABEL_PATH = 'models/class_labels_2.json'
TEMP_IMAGE_DIR = 'Data/Temp'

st.set_page_config(page_title="Voice Classifier", page_icon="🔊")

st.title("🔊 Voice Speaker Classification")
st.write("Upload a voice recording (.wav or .mp3) to identify the speaker.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_audio_path = tmp.name

    st.audio(temp_audio_path)

    with st.spinner("Processing audio..."):
        plot_spectrogram(temp_audio_path, save_dir=TEMP_IMAGE_DIR)
        speaker, confidence = predict_speaker_from_folder(TEMP_IMAGE_DIR, MODEL_PATH, LABEL_PATH)
        clear_folder(TEMP_IMAGE_DIR)

    if speaker:
        st.success(f"✅ Predicted Speaker: **{speaker}**")
        st.info(f"Confidence: **{confidence:.2f}**")
    else:
        st.error("❌ No spectrogram images found for prediction.")
