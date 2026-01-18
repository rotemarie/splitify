# dataset.py
import os
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiStemDataset(Dataset):
    """
    Dataset for MUSDB18-style multi-stem audio.

    Expected HDF5 structure:
        - mixture
        - vocals
        - drums
        - bass
        - other

    Returned sample (mono):
      {
        "mix": Tensor (1, L),
        "target": Tensor (n_stems, 1, L),
        "hdf5_path": "...",
        "start_sample": int,
        "end_sample": int,
        "stems": ["vocals", "drums", "bass", "other"]
      }
    """

    def __init__(
        self,
        index_pkl: str,
        mixture_key: str = "mixture",
        stems: List[str] = ["vocals", "drums", "bass", "other"],
        to_mono: bool = True,
        compute_accompaniment_if_missing: bool = True,
    ):
        """
        Args:
            index_pkl: path to the pickle file created by indexer.
            mixture_key: dataset name for the mixture.
            stems: list of stem keys to load.
            to_mono: convert all to mono (avg channels).
        """
        self.index_pkl = Path(index_pkl)
        if not self.index_pkl.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_pkl}")

        with open(self.index_pkl, "rb") as f:
            self.indexes = pickle.load(f)

        self.mixture_key = mixture_key
        self.stems = stems
        self.to_mono = to_mono
        self.compute_accompaniment_if_missing = compute_accompaniment_if_missing

        self._h5_cache: Dict[str, h5py.File] = {}

    def __len__(self):
        return len(self.indexes)

    def _open_h5(self, path: str) -> h5py.File:
        """Cache and reuse HDF5 handles."""
        if path in self._h5_cache and self._h5_cache[path].id:
            return self._h5_cache[path]
        hf = h5py.File(path, "r")
        self._h5_cache[path] = hf
        return hf

    def _read_segment(self, hf: h5py.File, key: str, start: int, end: int) -> np.ndarray:
        if key not in hf:
            raise KeyError(f"Key '{key}' not found in HDF5: {hf.filename}")
        data = hf[key][:, start:end].astype(np.float32)
        return data

    def _compute_accompaniment(self, hf: h5py.File, start: int, end: int) -> np.ndarray:
        """Compute accompaniment = mixture - vocals."""
        mix = self._read_segment(hf, self.mixture_key, start, end)
        vocals = self._read_segment(hf, "vocals", start, end)
        return (mix - vocals).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.indexes[idx]
        h5_path = entry["hdf5_path"]
        start = entry["start_sample"]
        end = entry["end_sample"]

        # Open HDF5 file
        hf = self._open_h5(h5_path)

        # Read mixture
        if self.mixture_key not in hf:
            raise KeyError(f"Mixture key '{self.mixture_key}' not found in {h5_path}")
        
        mixture = self._read_segment(hf, self.mixture_key, start, end)  # (C, L)
        
        # Read all target stems
        targets = {}
        for stem_name in self.stems:
            if stem_name in hf:
                targets[stem_name] = self._read_segment(hf, stem_name, start, end)
            elif stem_name == "other" and self.compute_accompaniment_if_missing:
                # Compute "other" as mixture - vocals - bass - drums
                other = mixture.copy()
                for other_stem in ["vocals", "bass", "drums"]:
                    if other_stem in hf:
                        other -= self._read_segment(hf, other_stem, start, end)
                targets[stem_name] = other
            else:
                # Fill with zeros if stem is missing
                targets[stem_name] = np.zeros_like(mixture)

        # Convert to mono if requested
        if self.to_mono:
            mixture = np.mean(mixture, axis=0, keepdims=True)  # (1, L)
            for stem_name in targets:
                targets[stem_name] = np.mean(targets[stem_name], axis=0, keepdims=True)

        # Convert to torch tensors
        mixture_tensor = torch.from_numpy(mixture).float()
        target_tensors = {k: torch.from_numpy(v).float() for k, v in targets.items()}

        return {
            "mixture": mixture_tensor,  # Shape: (C, L) - DataLoader will batch to (B, C, L)
            "targets": target_tensors,  # Dict of {stem: (C, L)}
            "hdf5_path": h5_path,
            "start_sample": start,
            "end_sample": end,
            "stems": self.stems
        }

    def close(self):
        """Close all cached HDF5 file handles."""
        for path, hf in list(self._h5_cache.items()):
            try:
                if hf.id:  # Check if file is still open
                    hf.close()
            except Exception:
                pass
            self._h5_cache.pop(path, None)

    def __del__(self):
        """Cleanup when dataset is garbage collected."""
        try:
            self.close()
        except Exception:
            pass


# Create alias for compatibility
AudioSegmentDataset = MultiStemDataset
