import os
import tempfile
import joblib
import librosa
import librosa.display
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# MUSIC GENRE CLASSIFICATION - CORRECTED STREAMLIT APP
# ============================================================
# IMPORTANT:
# The supplied features_3_sec.csv contains 58 model features.
# One of them is "length", and its value in the training data is
# CONSTANTLY 66149.
#
# The previous app filled "length" with 0. This is a serious
# train/prediction mismatch and can cause the SVM to predict the
# same genre repeatedly.
#
# This version sets length = 66149 and uses the saved feature
# order from the notebook.
# ============================================================

st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Genre Classification")
st.write(
    "Upload a song and use Librosa + the trained SVM model "
    "to predict its music genre."
)

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR, "music_genre_model.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR, "label_encoder.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_DIR, "feature_columns.pkl"
)

# ------------------------------------------------------------
# Check required files
# ------------------------------------------------------------

required_files = [
    (MODEL_PATH, "music_genre_model.pkl"),
    (ENCODER_PATH, "label_encoder.pkl"),
    (FEATURE_COLUMNS_PATH, "feature_columns.pkl"),
]

for path, filename in required_files:
    if not os.path.exists(path):
        st.error(
            f"{filename} was not found. "
            "Run the notebook from top to bottom first."
        )
        st.stop()


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

@st.cache_resource
def load_saved_objects():

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, encoder, feature_columns


model, encoder, feature_columns = load_saved_objects()

feature_columns = list(feature_columns)


# ------------------------------------------------------------
# Verify the saved model's expected feature count
# ------------------------------------------------------------

if len(feature_columns) != 58:

    st.warning(
        f"The saved notebook currently contains "
        f"{len(feature_columns)} features. "
        f"The supplied features_3_sec.csv uses 58 features."
    )


# ============================================================
# LIBROSA FEATURE EXTRACTION
# ============================================================

def extract_segment_features(y, sr):
    """
    Extract the feature structure used by features_3_sec.csv.

    The training notebook uses:
        length
        chroma_stft_mean/var
        rms_mean/var
        spectral_centroid_mean/var
        spectral_bandwidth_mean/var
        rolloff_mean/var
        zero_crossing_rate_mean/var
        harmony_mean/var
        perceptr_mean/var
        tempo
        mfcc1..mfcc20 mean/var
    """

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    rms = librosa.feature.rms(
        y=y
    )

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )
    )

    rolloff = (
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )
    )

    zero_crossing_rate = (
        librosa.feature.zero_crossing_rate(
            y
        )
    )

    harmony, percussive = (
        librosa.effects.hpss(y)
    )

    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    tempo = float(
        np.asarray(tempo).reshape(-1)[0]
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )


    # --------------------------------------------------------
    # Build dictionary
    # --------------------------------------------------------

    features = {

        # IMPORTANT:
        # The CSV has length = 66149 for every row.
        # Do NOT use 0 here.
        "length": 66149,

        "chroma_stft_mean":
            float(np.mean(chroma)),

        "chroma_stft_var":
            float(np.var(chroma)),

        "rms_mean":
            float(np.mean(rms)),

        "rms_var":
            float(np.var(rms)),

        "spectral_centroid_mean":
            float(np.mean(spectral_centroid)),

        "spectral_centroid_var":
            float(np.var(spectral_centroid)),

        "spectral_bandwidth_mean":
            float(np.mean(spectral_bandwidth)),

        "spectral_bandwidth_var":
            float(np.var(spectral_bandwidth)),

        "rolloff_mean":
            float(np.mean(rolloff)),

        "rolloff_var":
            float(np.var(rolloff)),

        "zero_crossing_rate_mean":
            float(np.mean(zero_crossing_rate)),

        "zero_crossing_rate_var":
            float(np.var(zero_crossing_rate)),

        "harmony_mean":
            float(np.mean(harmony)),

        "harmony_var":
            float(np.var(harmony)),

        "perceptr_mean":
            float(np.mean(percussive)),

        "perceptr_var":
            float(np.var(percussive)),

        "tempo":
            tempo
    }


    # --------------------------------------------------------
    # MFCC 1-20
    # --------------------------------------------------------

    for i in range(20):

        features[
            f"mfcc{i + 1}_mean"
        ] = float(
            np.mean(mfcc[i])
        )

        features[
            f"mfcc{i + 1}_var"
        ] = float(
            np.var(mfcc[i])
        )


    feature_df = pd.DataFrame(
        [features]
    )


    # --------------------------------------------------------
    # EXACT training column order
    # --------------------------------------------------------

    missing = [
        c for c in feature_columns
        if c not in feature_df.columns
    ]

    if missing:
        raise ValueError(
            "The Librosa feature extractor is missing "
            f"these training features: {missing}"
        )


    feature_df = feature_df[
        feature_columns
    ]


    return feature_df


# ============================================================
# EXTRACT 3-SECOND SEGMENTS
# ============================================================

def extract_song_features(
    audio_path,
    segment_seconds=3,
    sr=22050
):

    y, sr = librosa.load(
        audio_path,
        sr=sr,
        mono=True
    )

    if len(y) == 0:
        raise ValueError(
            "No readable audio was found."
        )


    segment_length = int(
        segment_seconds * sr
    )

    feature_rows = []


    for start in range(
        0,
        len(y),
        segment_length
    ):

        segment = y[
            start:start + segment_length
        ]


        # Ignore very short final fragment.
        if len(segment) < int(
            0.75 * segment_length
        ):
            continue


        row = extract_segment_features(
            segment,
            sr
        )

        feature_rows.append(row)


    if not feature_rows:
        raise ValueError(
            "The song is too short for prediction."
        )


    features = pd.concat(
        feature_rows,
        ignore_index=True
    )


    return features, y, sr


# ============================================================
# GENRE PREDICTION
# ============================================================

def predict_music_genre(audio_path):

    (
        features,
        audio,
        sr
    ) = extract_song_features(
        audio_path
    )


    # --------------------------------------------------------
    # Predict every 3-second segment
    # --------------------------------------------------------

    predictions = model.predict(
        features
    )


    # --------------------------------------------------------
    # Majority vote
    # --------------------------------------------------------

    labels, counts = np.unique(
        predictions,
        return_counts=True
    )

    winning_label = labels[
        np.argmax(counts)
    ]

    final_genre = encoder.inverse_transform(
        [winning_label]
    )[0]

    vote_percentage = (
        counts.max()
        /
        len(predictions)
    ) * 100


    # --------------------------------------------------------
    # Segment vote table
    # --------------------------------------------------------

    vote_table = pd.DataFrame({
        "Genre": encoder.inverse_transform(
            labels
        ),
        "Segments": counts
    })

    vote_table["Percentage"] = (
        vote_table["Segments"]
        /
        len(predictions)
        * 100
    )

    vote_table = vote_table.sort_values(
        "Segments",
        ascending=False
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Probability prediction
    # --------------------------------------------------------

    probability_table = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            features
        )

        mean_probabilities = (
            probabilities.mean(axis=0)
        )

        probability_table = pd.DataFrame({

            "Genre":
                encoder.inverse_transform(
                    np.arange(
                        len(mean_probabilities)
                    )
                ),

            "Probability":
                mean_probabilities

        }).sort_values(
            "Probability",
            ascending=False
        ).reset_index(
            drop=True
        )


    return (
        final_genre,
        vote_percentage,
        vote_table,
        probability_table,
        features,
        audio,
        sr
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🎧 Upload Music"
)

uploaded_file = st.sidebar.file_uploader(
    "Choose a music file",
    type=[
        "wav",
        "mp3",
        "ogg",
        "flac",
        "m4a"
    ]
)


# ============================================================
# APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a WAV, MP3, OGG, FLAC or M4A "
        "file to start prediction."
    )

    st.markdown(
        """
        ### Prediction pipeline

        **Music file**
        ↓

        **Librosa**
        ↓

        **3-second segments**
        ↓

        **58 matching audio features**
        ↓

        **Saved SVM model**
        ↓

        **Majority voting**
        ↓

        **Final music genre**
        """
    )


else:

    st.subheader(
        "🎵 Selected Song"
    )

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    st.audio(
        uploaded_file
    )


    if st.button(
        "🎯 Predict Music Genre",
        use_container_width=True
    ):

        temporary_file = None

        try:

            extension = os.path.splitext(
                uploaded_file.name
            )[1]


            # ----------------------------------------------
            # Save uploaded audio temporarily
            # ----------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
            ) as temp:

                temp.write(
                    uploaded_file.getbuffer()
                )

                temporary_file = temp.name


            # ----------------------------------------------
            # Predict
            # ----------------------------------------------

            with st.spinner(
                "Librosa is extracting features..."
            ):

                (
                    genre,
                    vote_percentage,
                    vote_table,
                    probability_table,
                    extracted_features,
                    audio,
                    sr
                ) = predict_music_genre(
                    temporary_file
                )


            # ----------------------------------------------
            # FINAL RESULT
            # ----------------------------------------------

            st.success(
                f"🎵 Predicted Genre: {genre.upper()}"
            )


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Genre",
                    genre.title()
                )

            with col2:

                st.metric(
                    "Segments Analyzed",
                    len(
                        extracted_features
                    )
                )


            st.info(
                f"The winning genre appeared in "
                f"{vote_percentage:.2f}% of the "
                f"3-second segment predictions."
            )


            # ==================================================
            # SEGMENT VOTING
            # ==================================================

            st.subheader(
                "🗳️ Segment Voting"
            )


            vote_display = vote_table.copy()

            vote_display["Percentage"] = (
                vote_display["Percentage"]
                .round(2)
            )


            st.dataframe(
                vote_display,
                use_container_width=True,
                hide_index=True
            )


            # ==================================================
            # PROBABILITY
            # ==================================================

            if probability_table is not None:

                st.subheader(
                    "📊 Prediction Confidence"
                )


                probability_display = (
                    probability_table.copy()
                )


                probability_display[
                    "Probability"
                ] = (
                    probability_display[
                        "Probability"
                    ] * 100
                ).round(2)


                probability_display = (
                    probability_display.rename(
                        columns={
                            "Probability":
                                "Confidence (%)"
                        }
                    )
                )


                st.dataframe(
                    probability_display,
                    use_container_width=True,
                    hide_index=True
                )


                st.subheader(
                    "📈 Genre Probability"
                )


                chart_data = (
                    probability_table
                    .set_index(
                        "Genre"
                    )["Probability"]
                    .sort_values(
                        ascending=False
                    )
                )


                st.bar_chart(
                    chart_data
                )


            # ==================================================
            # FEATURES
            # ==================================================

            with st.expander(
                "🔍 View Extracted Features"
            ):

                st.write(
                    "These are the features extracted "
                    "from the uploaded song using Librosa."
                )

                st.dataframe(
                    extracted_features,
                    use_container_width=True,
                    hide_index=True
                )


            # ==================================================
            # WAVEFORM
            # ==================================================

            st.subheader(
                "〰️ Audio Waveform"
            )


            fig = plt.figure(
                figsize=(12, 4)
            )


            librosa.display.waveshow(
                audio,
                sr=sr
            )


            plt.title(
                "Waveform of Uploaded Song"
            )

            plt.xlabel(
                "Time (seconds)"
            )

            plt.ylabel(
                "Amplitude"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


            # ==================================================
            # MEL SPECTROGRAM
            # ==================================================

            st.subheader(
                "🎼 Mel Spectrogram"
            )


            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=64
            )


            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )


            fig2 = plt.figure(
                figsize=(12, 5)
            )


            librosa.display.specshow(
                mel_db,
                sr=sr,
                x_axis="time",
                y_axis="mel"
            )


            plt.colorbar(
                format="%+2.0f dB"
            )


            plt.title(
                "Mel Spectrogram"
            )


            plt.tight_layout()

            st.pyplot(fig2)

            plt.close(fig2)


        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(error)


        finally:

            if (
                temporary_file
                and os.path.exists(
                    temporary_file
                )
            ):

                os.remove(
                    temporary_file
                )
