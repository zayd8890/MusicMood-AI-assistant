import pickle
import numpy as np
import faiss
import librosa
import openl3

class similar_songs:
    def __init__(self, audio_input):
        self.audio_input = audio_input
        self.vector_db = None
        self.metadata = None
        self.embedder_model = None

    def load_index(self, index_path, metadata_path):
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.vector_db = faiss.read_index(index_path)
        self.embedder_model = openl3.models.load_audio_embedding_model(
            input_repr="mel256", content_type="music", embedding_size=512
        )

    def extract_embedding(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30.0)
        target_samples = int(22050*30.0)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        emb, _ = openl3.get_audio_embedding(audio, sr, model=self.embedder_model, hop_size=1.0, verbose=0)
        emb = np.mean(emb, axis=0).astype("float32")
        emb /= np.linalg.norm(emb) + 1e-10
        return emb.reshape(1, -1)

    def main(self, k=10):
        if self.vector_db is None or self.metadata is None:
            return "⚠️ Please load index and metadata first"
        query_emb = self.extract_embedding(self.audio_input)
        distances, indices = self.vector_db.search(query_emb, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[idx]
            results.append({
                "title": meta["title"],
                "artist": meta.get("artist", "unknown"),
                "genre": meta.get("genre", "unknown"),
                "similarity_score": float(dist)
            })
        return results
