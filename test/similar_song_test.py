from feature.similar_songs import similar_songs
from pathlib import Path

TEST_AUDIO = Path("dataset/test_song.mp3")
INDEX_PATH = Path("indexes/music_index_openl3")
METADATA_PATH = Path("metadata/songs_metadata.pkl")

reco = similar_songs(TEST_AUDIO)
reco.load_index(INDEX_PATH, METADATA_PATH)

results = reco.main(k=10)

for i, r in enumerate(results, 1):
    print(f"{i}. {r['title']} | {r['artist']} | {r['genre']} | {r['similarity_score']:.3f}")
