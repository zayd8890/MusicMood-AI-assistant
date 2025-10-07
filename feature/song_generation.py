import re
import numpy as np
import soundfile as sf
import librosa
import torch
from IPython.display import Audio, display
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer

class SongGeneration:
    """
    SongGeneration class wraps MusicGen model and generates music from any text input.
    Supports multilingual input, optional melody, and GPU acceleration.
    """
    
    def __init__(self, user_text, num_versions=1, upload_audio_path=None, duration=15, sr=16000):
        """
        Initialize the song generator.
        """
        self.user_text = user_text
        self.num_versions = num_versions
        self.upload_audio_path = upload_audio_path
        self.duration = duration
        self.sr = sr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load models
        print(f"Using device: {self.device}")
        print("Loading MusicGen model...")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(self.device)
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        print("✅ MusicGen loaded.")

        print("Loading translation model...")
        self.translator_model_name = "Helsinki-NLP/opus-mt-mul-en"
        self.translator_tokenizer = MarianTokenizer.from_pretrained(self.translator_model_name)
        self.translator_model = MarianMTModel.from_pretrained(self.translator_model_name).to(self.device)
        print("✅ Translation model loaded.\n")

        # Presets
        self.keyword_presets = {
            "jazz": "smooth jazz instrumental",
            "hip hop": "energetic hip-hop track",
            "lofi": "chill lo-fi beat",
            "chill": "chill lo-fi beat",
            "sad": "melancholic piano solo",
            "happy": "happy upbeat pop song",
            "dark": "dark cinematic soundtrack",
            "pop": "happy upbeat pop song",
            "electronic": "electronic dance music",
            "classical": "classic orchestral theme",
            "ambient": "ambient relaxing atmosphere",
            "rock": "classic rock guitar riff",
            "funk": "groovy funk bassline",
            "epic": "epic fantasy adventure theme",
            "romantic": "soft romantic piano melody",
            "synthwave": "retro synthwave track",
            "blues": "classic blues guitar",
            "reggae": "chill reggae rhythm",
            "metal": "heavy metal guitar riff",
            "country": "country acoustic track",
            "techno": "driving techno beat",
            "folk": "acoustic folk song",
            "instrumental": "soft instrumental track",
            "trance": "uplifting trance track",
            "relaxing": "soothing ambient music",
            "energetic": "high-energy dance track",
            "cinematic": "epic cinematic score",
            "vocal": "soft vocal melody",
            "orchestral": "full orchestral arrangement",
            "groovy": "groovy funk rhythm",
            "mellow": "mellow jazz piano",
            "dreamy": "dreamy ambient soundscape",
            "dramatic": "dramatic orchestral theme",
            "motivational": "uplifting motivational track",
            "party": "energetic party dance music",
            "summer": "bright summer pop tune",
            "winter": "soft winter piano music",
            "rainy": "melancholic rainy day track",
            "uplifting": "happy uplifting melody",
            "darkwave": "dark electronic soundscape",
            "folkrock": "folk-rock acoustic band",
            "bluesrock": "blues rock guitar solo",
            "soul": "classic soul vocals",
            "rnb": "smooth R&B groove",
            "trap": "trap beat with pads",
            "vintage": "vintage jazz swing",
            "meditative": "calm meditative soundscape",
            "space": "space ambient journey",
            "holiday": "festive holiday music"
        }

    # ------------------------
    # Translation
    # ------------------------
    def translate_to_english(self, text):
        batch = self.translator_tokenizer([text], return_tensors="pt", padding=True).to(self.device)
        translated = self.translator_model.generate(**batch)
        en_text = self.translator_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        return en_text

    # ------------------------
    # Audio helpers
    # ------------------------
    def clean_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=22050, mono=True)
        audio = audio / np.max(np.abs(audio))
        return audio, sr

    def postprocess_audio(self, audio_array):
        audio_array = audio_array / np.max(np.abs(audio_array))
        desired_samples = self.sr * self.duration
        fade_len = 1024
        if len(audio_array) < desired_samples:
            audio_array = np.pad(audio_array, (0, desired_samples - len(audio_array)))
        else:
            audio_array = audio_array[:desired_samples]
        audio_array[:fade_len] *= np.linspace(0, 1, fade_len)
        audio_array[-fade_len:] *= np.linspace(1, 0, fade_len)
        return audio_array

    # ------------------------
    # Map user input → preset
    # ------------------------
    def map_to_preset(self, text):
        preset = text.lower()
        for key, value in self.keyword_presets.items():
            if key in preset:
                return value
        return preset

    # ------------------------
    # Main method
    # ------------------------
    def main(self):
        # Translate
        translated_text = self.translate_to_english(self.user_text)
        print(f"\n🌐 Translated to English: {translated_text}")

        # Preset selection
        preset = self.map_to_preset(translated_text)
        normalized_selection = re.sub(r"[^a-zA-Z0-9\s]", "", preset.strip().lower())
        print(f"\n🎵 Using preset: '{normalized_selection}'\n")

        # Generate music
        for i in range(self.num_versions):
            if self.upload_audio_path:
                melody, sr_melody = self.clean_audio(self.upload_audio_path)
                inputs = self.processor(
                    text=[normalized_selection],
                    audio=[melody],
                    sampling_rate=sr_melody,
                    return_tensors="pt"
                )
            else:
                inputs = self.processor(text=[normalized_selection], return_tensors="pt")

            # Move inputs to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate audio
            audio_values = self.model.generate(**inputs, max_new_tokens=512)
            audio_array = audio_values[0, 0].cpu().numpy()
            audio_array = self.postprocess_audio(audio_array)

            filename = f"output_{i+1}.wav" if self.num_versions > 1 else "output.wav"
            sf.write(filename, audio_array, samplerate=self.sr)
            print(f"✅ Music version {i+1} saved as: {filename}")

            display(Audio(audio_array, rate=self.sr))

        print("\n🎶 Done! All generated files are saved.")
