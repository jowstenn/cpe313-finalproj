import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import tempfile
import matplotlib.pyplot as plt

st.set_page_config(page_title="Urban Noise Classifier", layout="centered")

# load model and encoder
@st.cache_resource
def load_mel_model():
    return tf.keras.models.load_model("mel_model4.h5")

@st.cache_resource
def load_mfcc_model():
    return tf.keras.models.load_model("mfcc_model.h5")

@st.cache_resource
def load_mel_encoder():
    with open("label_encoder2.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_mfcc_encoder():
    with open("label_encoder_mfcc.pkl", "rb") as f:
        return pickle.load(f)

mel_model = load_mel_model()
mfcc_model = load_mfcc_model()

mel_le = load_mel_encoder()
mfcc_le = load_mfcc_encoder()


# feature extraction (must be same with colab)
def extract_mel(file_path, max_len=128):
    y, sr = librosa.load(file_path, sr=22050, duration=4)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel)

    # Fix size
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad_width)))
    else:
        mel_db = mel_db[:, :max_len]

    return mel_db

def extract_mfcc(file_path, max_len=128):
    y, sr = librosa.load(file_path, sr=22050, duration=4)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


st.title("Urban Noise Classification")
st.write("Upload a WAV file and choose a model approach.")

model_option = st.selectbox(
    "Choose Feature Extraction Method",
    ["Mel Spectrogram", "MFCC"]
)

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav"]
)

if uploaded_file is not None:

    # play audio
    st.audio(uploaded_file, format="audio/wav")

    # save temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    
    # mel spectrogram model
    if model_option == "Mel Spectrogram":

        feature = extract_mel(temp_path)

        X_input = feature[np.newaxis, ..., np.newaxis]

        prediction = mel_model.predict(X_input)
        probs = prediction[0]

        class_index = np.argmax(probs)
        class_label = mel_le.inverse_transform([class_index])[0]
        class_names = mel_le.classes_

        confidence = np.max(probs)

        st.success(f"Predicted Class: {class_label}")
        st.info(f"Confidence: {confidence:.2%}")

        # visualization
        st.subheader("Mel Spectrogram")

        fig, ax = plt.subplots(figsize=(10, 4))
        img = ax.imshow(feature, aspect="auto", origin="lower")
        ax.set_title("Mel Spectrogram")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel Bands")
        plt.colorbar(img, ax=ax)

        st.pyplot(fig)

    
    # mfcc model
    else:

        feature = extract_mfcc(temp_path)

        X_input = feature[np.newaxis, ..., np.newaxis]

        prediction = mfcc_model.predict(X_input)
        probs = prediction[0]

        class_index = np.argmax(probs)
        class_label = mfcc_le.inverse_transform([class_index])[0]
        class_names = mfcc_le.classes_

        confidence = np.max(probs)

        st.success(f"Predicted Class: {class_label}")
        st.info(f"Confidence: {confidence:.2%}")

        # visualization
        st.subheader("MFCC")

        fig, ax = plt.subplots(figsize=(10, 4))
        img = ax.imshow(feature, aspect="auto", origin="lower")
        ax.set_title("MFCC")
        ax.set_xlabel("Time")
        ax.set_ylabel("Coefficients")
        plt.colorbar(img, ax=ax)

        st.pyplot(fig)

    
    # top 3 predictions
    st.subheader("Top Predictions")

    top3_idx = np.argsort(probs)[-3:][::-1]

    for i in top3_idx:
        st.write(f"{class_names[i]} : {probs[i]:.2%}")

    # bar chart
    st.subheader("All Class Probabilities")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(class_names, probs)
    ax2.set_xticklabels(class_names, rotation=45, ha="right")
    ax2.set_ylabel("Probability")
    ax2.set_title("Prediction Confidence")

    st.pyplot(fig2)