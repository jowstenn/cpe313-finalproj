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
def load_model():
    return tf.keras.models.load_model("mel_model2.h5")

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
le = load_encoder()

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

st.title("Urban Noise Classification")
st.subheader("Mel Spectrogram + CNN Model")

st.write("Upload a .wav audio file to classify urban sounds.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:

    # play audio
    st.audio(uploaded_file, format="audio/wav")

    # save temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # extract feature
    feature = extract_mel(temp_path)
    feature_input = feature[np.newaxis, ..., np.newaxis]

    # predict
    prediction = model.predict(feature_input)
    class_index = np.argmax(prediction)
    class_label = le.inverse_transform([class_index])[0]
    confidence = np.max(prediction)

    # display results
    st.success(f"Predicted Class: {class_label}")
    st.info(f"Confidence: {confidence:.2f}")

    # visualization of spectrogram
    st.subheader("Mel Spectrogram Visualization")

    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(feature, aspect='auto', origin='lower')
    ax.set_title("Mel Spectrogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel Frequency")

    plt.colorbar(img, ax=ax)

    st.pyplot(fig)

    # probability distribution
    st.subheader("Prediction Probabilities")

    probs = prediction[0]
    class_names = le.classes_

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(class_names, probs)
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.set_ylabel("Probability")
    ax2.set_title("Class Probabilities")

    st.pyplot(fig2)
