import os
import tempfile
import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join("models", "music_genre_model.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
FEATURE_COLUMNS_PATH = os.path.join("models", "feature_columns.pkl")

@st.cache_resource
def load_model():
    return (
        joblib.load(MODEL_PATH),
        joblib.load(ENCODER_PATH),
        joblib.load(FEATURE_COLUMNS_PATH)
    )

def extract_segment_features(y, sr, feature_columns):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony, percussive = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).reshape(-1)[0])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    d = {
        "chroma_stft_mean": float(np.mean(chroma)),
        "chroma_stft_var": float(np.var(chroma)),
        "rms_mean": float(np.mean(rms)),
        "rms_var": float(np.var(rms)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_var": float(np.var(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_var": float(np.var(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_var": float(np.var(rolloff)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "zero_crossing_rate_var": float(np.var(zcr)),
        "harmony_mean": float(np.mean(harmony)),
        "harmony_var": float(np.var(harmony)),
        "perceptr_mean": float(np.mean(percussive)),
        "perceptr_var": float(np.var(percussive)),
        "tempo": tempo,
    }
    for i in range(20):
        d[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        d[f"mfcc{i+1}_var"] = float(np.var(mfcc[i]))

    return pd.DataFrame([d]).reindex(columns=feature_columns, fill_value=0)

def predict_song(audio_path, model, encoder, feature_columns):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    segment_length = int(3 * sr)
    rows = []

    for start in range(0, len(y), segment_length):
        segment = y[start:start+segment_length]
        if len(segment) < int(0.75 * segment_length):
            continue
        rows.append(extract_segment_features(segment, sr, feature_columns))

    if not rows:
        raise ValueError("Audio file is too short for prediction.")

    X_new = pd.concat(rows, ignore_index=True)
    preds = model.predict(X_new)

    values, counts = np.unique(preds, return_counts=True)
    final_pred = values[np.argmax(counts)]
    genre = encoder.inverse_transform([final_pred])[0]

    probability_df = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_new).mean(axis=0)
        probability_df = pd.DataFrame({
            "Genre": encoder.inverse_transform(np.arange(len(probs))),
            "Probability": probs
        }).sort_values("Probability", ascending=False)

    return genre, probability_df, X_new, y, sr

st.set_page_config(page_title="Music Genre Classification", page_icon="🎵")
st.markdown("<h1 style='text-align:center;'>Music Genre Classification</h1>", unsafe_allow_html=True)
st.write("Upload a song. Librosa extracts the same audio-feature structure used during training, then the trained best classifier predicts the genre.")

uploaded_file = st.file_uploader(
    "Upload a music file",
    type=["wav", "mp3", "ogg", "flac", "m4a"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Predict Genre"):
        temp_path = None
        try:
            model, encoder, feature_columns = load_model()
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded_file.getbuffer())
                temp_path = f.name

            genre, probability_df, feature_data, y, sr = predict_song(
                temp_path, model, encoder, feature_columns
            )

            st.success(f"Predicted Genre: {genre.title()}")

            if probability_df is not None:
                st.subheader("Prediction Confidence")
                st.dataframe(
                    probability_df.style.format({"Probability": "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True
                )

            with st.expander("View extracted features"):
                st.dataframe(feature_data, use_container_width=True)

            st.subheader("Waveform")
            st.line_chart(pd.Series(y))

        except Exception as e:
            st.error("Could not process this audio file. Try a standard WAV/MP3/OGG/FLAC file.")
            st.exception(e)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
