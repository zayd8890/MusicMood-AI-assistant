import faiss
import pickle
import openl3
import librosa
import numpy as np
from pathlib import Path

class similar_songs: 
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.index = None
        self.metadata = None
        self.model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=512)

    def load_index(self, index_path, metadata_path):
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)

        # Handle the wrapped structure
        if isinstance(data, dict) and "metadata" in data:
            self.metadata = data["metadata"]
        else:
            self.metadata = data


    def extract_embedding(self, filepath):
        audio, sr = librosa.load(filepath, sr=22050, mono=True, duration=30.0)
        if len(audio) < 22050*30:
            audio = np.pad(audio, (0, 22050*30 - len(audio)))
        emb, _ = openl3.get_audio_embedding(audio, sr, model=self.model, hop_size=1.0, verbose=0)
        emb = np.mean(emb, axis=0).astype("float32")
        emb /= np.linalg.norm(emb) + 1e-10
        return emb

    def main(self, k=10):
        query_emb = self.extract_embedding(self.audio_path).reshape(1, -1)
        distances, indices = self.index.search(query_emb, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            m = self.metadata[idx]
            results.append({
                "title": m["title"],
                "artist": m["artist"],
                "genre": m["genre"],
                "filepath": m["filepath"],
                "similarity_score": float(dist)
            })
        return results
