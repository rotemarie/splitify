import h5py
import yaml
from pathlib import Path
import pickle


def create_indexes(workspace: str, config_yaml: str, split: str = "train"):
    """Create segment-based indexes from hdf5 dataset."""
    # Load config
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    sample_rate = config["sample_rate"]
    segment_seconds = config["segment_seconds"]

    # Navigate to split section (train or test)
    split_cfg = config.get(split)
    if split_cfg is None:
        raise KeyError(f"No split '{split}' found in {config_yaml}")

    source_types = split_cfg["source_types"]

    # Output index directory
    #index_dir = Path(workspace) / split_cfg["indexes"]
    #index_dir.parent.mkdir(parents=True, exist_ok=True)
    index_file = Path(workspace) / config[split]["indexes"]
    index_file.parent.mkdir(parents=True, exist_ok=True)

    indexes = []

    # Loop through sources (vocals, bass, drums, etc.)
    for source_name, dataset_cfg in source_types.items():
        hdf5_dir = Path(workspace) / dataset_cfg["musdb18"]["hdf5s_directory"]
        key_in_hdf5 = dataset_cfg["musdb18"]["key_in_hdf5"]
        hop_seconds = dataset_cfg["musdb18"]["hop_seconds"]

        print(f"Processing source: {source_name}, hdf5_dir={hdf5_dir}")

        for hdf5_path in sorted(hdf5_dir.glob("*.h5")):
            with h5py.File(hdf5_path, "r") as hf:
                if key_in_hdf5 not in hf:
                    print(f"⚠️ Missing key {key_in_hdf5} in {hdf5_path}, skipping")
                    continue

                audio = hf[key_in_hdf5][:]
                total_samples = audio.shape[1]  # shape (channels, samples)
                segment_samples = int(segment_seconds * sample_rate)
                hop_samples = int(hop_seconds * sample_rate)

                for start in range(0, total_samples - segment_samples, hop_samples):
                    end = start + segment_samples
                    entry = {
                        "hdf5_path": str(hdf5_path),
                        "start_sample": start,
                        "end_sample": end,
                        "source": source_name,
                        "key": key_in_hdf5
                    }
                    indexes.append(entry)

    # Save index file (json instead of .pkl for readability)
    #index_file = Path(workspace) / split_cfg["indexes"]
    #with open(index_file, "w") as f:
    #    json.dump(indexes, f, indent=2)

    #index_file = index_dir / f"sr={sample_rate},{'-'.join(source_types)}.pkl"

    with open(index_file, "wb") as f:
        pickle.dump(indexes, f)

    print(f"✅ Saved {len(indexes)} segments to {index_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train")
    args = parser.parse_args()

    create_indexes(args.workspace, args.config_yaml, args.split)



'''
workspace = "/Users/rotemar/Documents/Splitify"
config_yaml = "/Users/rotemar/Documents/Splitify/preprocessing/configs/vocals-bass-drums-other,sr=44100,chn=2.yaml"
split = "train" #run once for test and one for train

run:
full_train:
python3 preprocessing/index.py \
  --workspace /Users/rotemar/Documents/Splitify \
  --config_yaml /Users/rotemar/Documents/Splitify/preprocessing/configs/vocals-bass-drums-other,sr=44100,chn=2.yaml \
  --split train
  

train:
python3 preprocessing/index.py \
  --workspace /Users/rotemar/Documents/Splitify \
  --config_yaml /Users/rotemar/Documents/Splitify/preprocessing/configs/sr=44100,vocals-bass-drums-other.yaml \
  --split train
  

'''