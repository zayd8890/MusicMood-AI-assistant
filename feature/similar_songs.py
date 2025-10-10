import faiss
import pickle
import openl3
import librosa
import numpy as np
from pathlib import Path

class similar_songs:
    def __init__(self, audio_input, sr=None):
        """
        audio_input: peut être soit un chemin de fichier (str ou Path),
                     soit un signal audio numpy array
        sr: taux d'échantillonnage (requis si audio_input est un array)
        """
        self.audio_input = audio_input
        self.sr = sr
        self.index = None
        self.metadata = None
        self.model = openl3.models.load_audio_embedding_model(
            input_repr="mel256", content_type="music", embedding_size=512
        )

    def load_index(self, index_path, metadata_path):
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)

        # handle wrapped dicts
        if isinstance(data, dict) and "metadata" in data:
            self.metadata = data["metadata"]
        else:
            self.metadata = data

    def extract_embedding(self):
        # Si audio_input est un chemin → charger le fichier
        if isinstance(self.audio_input, (str, Path)):
            audio, sr = librosa.load(self.audio_input, sr=22050, mono=True, duration=30.0)
        else:
            # Sinon, c’est un array déjà chargé
            audio, sr = self.audio_input, self.sr

        # Normaliser la durée
        target_len = 22050 * 30
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # Extraire l'embedding
        emb, _ = openl3.get_audio_embedding(audio, sr, model=self.model, hop_size=1.0, verbose=0)
        emb = np.mean(emb, axis=0).astype("float32")
        emb /= np.linalg.norm(emb) + 1e-10
        return emb

    def main(self, k=10):
        query_emb = self.extract_embedding().reshape(1, -1)
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
