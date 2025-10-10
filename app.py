import streamlit as st
import librosa
import numpy as np
from pathlib import Path
from feature.similar_songs import similar_songs

# -----------------------------
# 🎵 Page Title
# -----------------------------
st.title("🎶 Music Mood AI Assistant")

# Paths
INDEX_PATH = Path("indexes/music_index_openl3.faiss")
METADATA_PATH = Path("indexes/music_index_openl3.pkl")

# -----------------------------
# 🎧 Upload Audio
# -----------------------------
uploaded_file = st.file_uploader("Charge un extrait audio", type=["mp3", "wav"])

if uploaded_file is not None:
    # Show the audio player
    st.audio(uploaded_file, format="audio/wav")

    # Load audio file into numpy array
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)

    # Initialize recommender
    reco = similar_songs(y, sr)
    reco.load_index(INDEX_PATH, METADATA_PATH)

    # Run the search
    results = reco.main(k=10)

    # Show results
    st.subheader("🎵 Chansons similaires")
    for i, r in enumerate(results, 1):
        st.write(f"{i}. {r['title']} | {r['artist']} | {r['genre']} | Similarité: {r['similarity_score']:.3f}")
