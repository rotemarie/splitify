import os
import json
import argparse
import h5py
import numpy as np
import stempeg


def load_stems(path, sample_rate, channels):
    """Load .stem.mp4 file into separate stems."""
    audio, sr = stempeg.read_stems(path, sample_rate=sample_rate)

    if sr != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz but got {sr} Hz")

    # audio shape: (num_stems, num_samples, channels)
    stems = {}
    stem_names = ["mixture", "vocals", "drums", "bass", "other"]

    for i, name in enumerate(stem_names):
        if i < audio.shape[0]:
            stem_audio = audio[i]

            # Handle channels
            if stem_audio.shape[1] > channels:
                stem_audio = stem_audio[:, :channels]
            elif stem_audio.shape[1] < channels:
                stem_audio = np.tile(stem_audio, (1, channels))

            # Save as (channels, samples) instead of (samples, channels)
            stems[name] = stem_audio.T.astype("float32")

    return stems


def pack_subset_to_hdf5(dataset_dir, subset, hdf5_dir, sample_rate, channels):
    """Pack MUSDB18 subset into HDF5 files (one per track)."""
    os.makedirs(hdf5_dir, exist_ok=True)
    subset_dir = os.path.join(dataset_dir, subset)

    # 🔑 Fix: scan for `.stem.mp4` files, not folders
    tracks = [f for f in os.listdir(subset_dir) if f.endswith(".stem.mp4")]
    print(f"Found {len(tracks)} tracks in subset '{subset}'")

    for stem_file in tracks:
        track_name = os.path.splitext(os.path.splitext(stem_file)[0])[0]  # remove .stem.mp4
        stem_path = os.path.join(subset_dir, stem_file)
        hdf5_path = os.path.join(hdf5_dir, f"{track_name}.h5")

        stems = load_stems(stem_path, sample_rate, channels)

        with h5py.File(hdf5_path, "w") as hf:
            for stem_name, audio in stems.items():
                hf.create_dataset(stem_name, data=audio, dtype="float32", compression="gzip")

        print(f"✅ Packed {track_name} → {hdf5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack MUSDB18 stem.mp4s into HDF5s.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    dataset_dir = config["musdb18_dataset_dir"]
    workspace = config["workspace"]
    sr = config["sample_rate"]
    chn = config["channels"]

    for subset in config["subsets"]:
        hdf5s_dir = os.path.join(workspace, f"hdf5s/musdb18/sr={sr},chn={chn}/{subset}")
        pack_subset_to_hdf5(dataset_dir, subset, hdf5s_dir, sr, chn)


"""
Run like:
python3 preprocessing/audio_to_hdf5.py --config preprocessing/config.json
"""
