import pickle
from pathlib import Path

metadata_path = Path(r"indexes/music_index_openl3.pkl")

with open(metadata_path, "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict) and "metadata" in data:
    metadata = data["metadata"]
    print(f"Metadata entries found: {len(metadata)}")
else:
    print(f"Unexpected structure: {type(data)}")
    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
