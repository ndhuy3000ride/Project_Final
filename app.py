import streamlit as st
import os
import tempfile
from voice_utils import (
    plot_spectrogram,
    predict_speaker_from_folder,
    clear_folder,
    remove_silence_and_save
)

ENTROPY_THRESHOLD = 0.35
# ==== Định nghĩa mô hình ====
MODEL_OPTIONS = {
    "VGG16": {
        "model_path": "checkpoints/vgg16_model.h5",
        "label_path": "checkpoints/class_labels.json",
        "image_size": (224, 224)
    },
    "Custom CNN": {
        "model_path": "checkpoints/cnn_model_final.h5",
        "label_path": "checkpoints/class_labels.json",
        "image_size": (128, 128)
    },
    "Vision Transformer (ViT)": {
        "model_path": "checkpoints/final_vit_model.h5",
        "label_path": "checkpoints/class_labels.json",
        "image_size": (128, 128)
    },
    "ResNet50": {
        "model_path": "checkpoints/resnet50_model.h5",
        "label_path": "checkpoints/class_labels.json",
        "image_size": (224, 224)
    },
    "MobileNetV2": {
        "model_path": "checkpoints/mobilenetv2_model.h5",
        "label_path": "checkpoints/class_labels.json",
        "image_size": (128, 128)
    },
    "CNN - LSTM": {
        "model_path": "checkpoints/cnn_lstm_model_final.h5",
        "label_path": "checkpoints/class_labels.json",
        "image_size": (128, 128)
    },
    "Custom CNN - VGG16 Fusion": {
        "model_path": "checkpoints/vgg16_cnn_fusion_finetuned.h5",
        "label_path": "checkpoints/class_labels.json",
        "image_size": (128, 128)
    }
}

# ==== Giao diện ====
st.set_page_config(page_title="Voice Classifier", page_icon="🔊")

st.title("🔊 Voice Speaker Classification")
st.write("Upload a voice recording (.wav or .mp3) to identify the speaker.")

# Sidebar: chọn mô hình
st.sidebar.header("🧠 Choose Model")
selected_model_name = st.sidebar.selectbox("Select a model", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[selected_model_name]
MODEL_PATH = selected_model["model_path"]
LABEL_PATH = selected_model["label_path"]
IMAGE_SIZE = selected_model["image_size"]

TEMP_IMAGE_DIR = 'Data/Temp'

# Upload file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        temp_audio_path = tmp.name

    denoised_audio_path = temp_audio_path.replace(ext, "_clean.wav")
    st.audio(temp_audio_path)

    with st.spinner("Processing audio..."):
        remove_silence_and_save(temp_audio_path, denoised_audio_path)
        plot_spectrogram(denoised_audio_path, save_dir=TEMP_IMAGE_DIR)
        # plot_spectrogram(temp_audio_path, save_dir=TEMP_IMAGE_DIR)

        speaker, confidence, closest_speaker = predict_speaker_from_folder(
            TEMP_IMAGE_DIR, MODEL_PATH, LABEL_PATH, IMAGE_SIZE, ENTROPY_THRESHOLD
        )

        clear_folder(TEMP_IMAGE_DIR)

    if speaker:
        if speaker == "Unknown Speaker":
            st.warning(
                f"❓ Predicted: **Unknown Speaker** (Closest: **{closest_speaker}**, Confidence: {confidence:.2f})"
            )
            st.info(f"The system is not confident enough. The most likely match is: **{closest_speaker}**.")
        else:
            st.success(f"✅ Predicted Speaker: **{speaker}**")
            st.info(f"Confidence: **{confidence:.2f}**")
    else:
        st.error("❌ No spectrogram images found for prediction.")