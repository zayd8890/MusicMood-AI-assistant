# -----------------------------------------
# Imports
# -----------------------------------------
import os
import base64
import hmac
import hashlib
import time
import requests
from dataclasses import dataclass
from typing import Optional, Dict
from dotenv import load_dotenv
from pydub import AudioSegment
import io

# -----------------------------------------
# Data Models
# -----------------------------------------
@dataclass
class TrackMatch:
    title: Optional[str]
    artist: Optional[str]
    album: Optional[str] = None
    year: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None
    ids: Optional[Dict[str, str]] = None
    links: Optional[Dict[str, str]] = None
    artwork_url: Optional[str] = None


@dataclass
class LyricsResult:
    title: Optional[str]
    artist: Optional[str]
    url: Optional[str] = None
    snippet: Optional[str] = None


# -----------------------------------------
# Class principale : MusicRecognizer
# -----------------------------------------
class MusicRecognizer:
    '''
    Cette classe permet d’identifier une chanson à partir d’un extrait audio
    en utilisant l’API ACRCloud.
    '''

    def __init__(self):
        load_dotenv()
        self.ACR_HOST = os.getenv("ACR_HOST")
        self.ACR_ACCESS_KEY = os.getenv("ACR_ACCESS_KEY")
        self.ACR_ACCESS_SECRET = os.getenv("ACR_ACCESS_SECRET")

    # --- Génération de signature pour ACRCloud ---
    def _sign(self, method, uri, access_key, data_type, signature_version, timestamp, access_secret):
        string_to_sign = f"{method}\n{uri}\n{access_key}\n{data_type}\n{signature_version}\n{timestamp}"
        return base64.b64encode(
            hmac.new(access_secret.encode(), string_to_sign.encode(), hashlib.sha1).digest()
        ).decode()

    # --- Identification via ACRCloud ---
    def recognize_audio(self, audio_path: str, min_confidence: float = 65.0) -> TrackMatch:
        if not all([self.ACR_HOST, self.ACR_ACCESS_KEY, self.ACR_ACCESS_SECRET]):
            return TrackMatch(title=None, artist=None, score=0.0, source="demo")

        # Lecture du fichier audio
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Conversion en 15s max
        trimmed = self._trim_audio(audio_bytes)

        timestamp = str(int(time.time()))
        signature = self._sign("POST", "/v1/identify", self.ACR_ACCESS_KEY,
                               "audio", "1", timestamp, self.ACR_ACCESS_SECRET)
        files = {"sample": ("sample", trimmed, "audio/mpeg")}
        data = {
            "access_key": self.ACR_ACCESS_KEY,
            "sample_bytes": str(len(trimmed)),
            "timestamp": timestamp,
            "signature": signature,
            "data_type": "audio",
            "signature_version": "1",
        }

        url = f"https://{self.ACR_HOST}/v1/identify"
        resp = requests.post(url, files=files, data=data, timeout=10)

        if resp.status_code != 200:
            return TrackMatch(title=None, artist=None, score=0.0, source="error")

        payload = resp.json()
        parsed = self._parse_response(payload)

        if parsed and (parsed.get("score") or 0) >= min_confidence:
            tm = TrackMatch(
                title=parsed.get("title"),
                artist=parsed.get("artist"),
                album=parsed.get("album"),
                year=parsed.get("year"),
                score=parsed.get("score"),
                source="acrcloud",
                ids=parsed.get("ids"),
                links={}
            )
            if tm.ids and tm.ids.get("spotify"):
                tm.links["spotify"] = f"https://open.spotify.com/track/{tm.ids['spotify']}"
            if tm.ids and tm.ids.get("youtube"):
                tm.links["youtube"] = f"https://www.youtube.com/watch?v={tm.ids['youtube']}"
            return tm

        return TrackMatch(title=None, artist=None, score=0.0, source="demo")

    # --- Parse la réponse JSON d'ACRCloud ---
    def _parse_response(self, payload):
        try:
            if payload.get("status", {}).get("code") != 0:
                return None
            musics = payload.get("metadata", {}).get("music", [])
            if not musics:
                return None
            best = musics[0]
            title = best.get("title")
            artist = best.get("artists")[0]["name"] if best.get("artists") else None
            album = (best.get("album") or {}).get("name")
            rel = best.get("release_date") or ""
            year = rel[:4] if rel else None
            score = best.get("score")
            ext = best.get("external_metadata") or {}
            spotify_id = ((ext.get("spotify") or {}).get("track") or {}).get("id")
            youtube_id = (ext.get("youtube") or {}).get("vid")
            return {
                "title": title, "artist": artist, "album": album,
                "year": year, "score": score,
                "ids": {"spotify": spotify_id, "youtube": youtube_id}
            }
        except Exception:
            return None

    # --- Coupe un audio trop long ---
    def _trim_audio(self, uploaded_bytes: bytes, max_seconds: int = 15) -> bytes:
        try:
            audio = AudioSegment.from_file(io.BytesIO(uploaded_bytes))
            if len(audio) > max_seconds * 1000:
                audio = audio[:max_seconds * 1000]
            buf = io.BytesIO()
            audio.export(buf, format="mp3", bitrate="128k")
            return buf.getvalue()
        except Exception:
            return uploaded_bytes

    # --- Point d’entrée principal ---
    def main(self, audio_path: str):
        result = self.recognize_audio(audio_path)
        if result.title:
            print(f"🎵 {result.title} — {result.artist}")
            if result.links:
                for k, v in result.links.items():
                    print(f"{k.title()} link: {v}")
        else:
            print("⚠️ Aucun résultat trouvé ou clé API manquante.")
