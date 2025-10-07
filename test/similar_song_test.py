from feature.similar_songs import similar_songs
import sys

from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from feature.similar_songs import similar_songs


# paths
INDEX_PATH = Path("indexes/music_index_openl3")
METADATA_PATH = Path("metadata/songs_metadata.pkl")
TEST_AUDIO = "test_songs/my_song.mp3"

# create object
reco = similar_songs(TEST_AUDIO)

# load index
reco.load_index(INDEX_PATH, METADATA_PATH)

# get recommendations
results = reco.main(k=10)

# print results
for i, r in enumerate(results, 1):
    print(f"{i}. {r['title']} | {r['artist']} | {r['genre']} | similarity: {r['similarity_score']:.3f}")
