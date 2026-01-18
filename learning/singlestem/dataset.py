# dataset.py
import os
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class AudioSegmentDataset(Dataset):
    """
    Dataset for indexed HDF5 segments produced by your indexer.

    Expected index (list of dicts) entry formats (supported):
      - {"hdf5_path": "...", "start_sample": 0, "end_sample": N, "key": "vocals", "source": "vocals"}
      - {"hdf5_path": "...", "start_sample": 0, "end_sample": N, "sources": ["vocals","bass",...]}
      - {"hdf5_path": "...", "start_sample": 0, "end_sample": N, "key": "accompaniment", ...}

    The HDF5 files must contain at least the "mixture" dataset (channels x samples).
    Targets are read from the dataset key(s) specified in the index entry (entry["key"])
    or from the datasets listed in entry["sources"].

    If an entry requests "accompaniment" but the HDF5 has no "accompaniment" dataset,
    dataset will compute accompaniment = mixture - vocals (if vocals exists).

    Returned sample (dict):
      {
        "mix": Tensor (C, L) float32,
        "target": Tensor (C, L) float32,     # or stacked targets (n_targets, C, L) if multiple keys
        "hdf5_path": "...",
        "start_sample": int,
        "end_sample": int,
        "source": str (if present in entry)
      }
    """

    def __init__(
        self,
        index_pkl: str,
        mixture_key: str = "mixture",
        to_mono: bool = False,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        compute_accompaniment_if_missing: bool = True,
    ):
        """
        Args:
            index_pkl: path to the pickle file created by create_indexes (list of dicts).
            mixture_key: dataset name inside HDF5 for the mixture (default "mixture").
            to_mono: if True, average channels to mono (returned tensors will be shape (1, L)).
            transform: optional callable applied to each sample dict (mix/target tensors).
            compute_accompaniment_if_missing: if True, compute accompaniment = mixture - vocals
        """
        self.index_pkl = Path(index_pkl)
        if not self.index_pkl.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_pkl}")

        with open(self.index_pkl, "rb") as f:
            self.indexes = pickle.load(f)

        if not isinstance(self.indexes, list):
            raise ValueError("Index pickle must contain a list of index entries (dicts).")

        self.mixture_key = mixture_key
        self.to_mono = to_mono
        self.transform = transform
        self.compute_accompaniment_if_missing = compute_accompaniment_if_missing

        # cache for open h5 files in this process/worker
        self._h5_cache: Dict[str, h5py.File] = {}

    def __len__(self):
        return len(self.indexes)

    def _open_h5(self, path: str) -> h5py.File:
        """Open HDF5 file and cache handle (per process)."""
        if path in self._h5_cache:
            f = self._h5_cache[path]
            # reopen if closed unexpectedly
            if isinstance(f, h5py.File) and f.id:
                return f
        # Open read-only
        f = h5py.File(path, "r")
        self._h5_cache[path] = f
        return f

    def _read_dataset_segment(self, hf: h5py.File, key: str, start: int, end: int) -> np.ndarray:
        """Read dataset `key` from h5 file and return numpy array shaped (channels, length)."""
        if key not in hf:
            raise KeyError(f"Key '{key}' not found in HDF5 file: {hf.filename}")
        data = hf[key]
        # data stored as (channels, samples) expected
        # h5py slicing returns numpy array
        seg = data[:, start:end]
        return np.asarray(seg, dtype=np.float32)

    def _compute_accompaniment(self, hf: h5py.File, start: int, end: int) -> np.ndarray:
        """
        Compute accompaniment = mixture - vocals.
        Assumes both 'mixture' and 'vocals' exist.
        """
        mix = self._read_dataset_segment(hf, self.mixture_key, start, end)
        vocals = self._read_dataset_segment(hf, "vocals", start, end)
        # shape: (channels, samples)
        return (mix - vocals).astype(np.float32)

    def _get_target_arrays(self, hf: h5py.File, entry: dict) -> np.ndarray:
        """
        Return numpy array for target.
        If index entry defines multiple sources (entry['sources']), we stack them in axis=0:
            returns shape (n_sources, channels, samples)
        Otherwise returns shape (channels, samples)
        """
        # Case 1: explicit 'key'
        if "key" in entry:
            key = entry["key"]
            if key in hf:
                return self._read_dataset_segment(hf, key, entry["start_sample"], entry["end_sample"])
            elif key == "accompaniment" and self.compute_accompaniment_if_missing:
                # compute
                return self._compute_accompaniment(hf, entry["start_sample"], entry["end_sample"])
            else:
                raise KeyError(f"Target key '{key}' not in HDF5 and cannot be computed: {hf.filename}")

        # Case 2: 'sources' list provided (multiple targets)
        if "sources" in entry and isinstance(entry["sources"], (list, tuple)):
            arrays = []
            for sname in entry["sources"]:
                if sname in hf:
                    arr = self._read_dataset_segment(hf, sname, entry["start_sample"], entry["end_sample"])
                    arrays.append(arr)
                elif sname == "accompaniment" and self.compute_accompaniment_if_missing:
                    arr = self._compute_accompaniment(hf, entry["start_sample"], entry["end_sample"])
                    arrays.append(arr)
                else:
                    raise KeyError(f"Source '{sname}' not found in HDF5 {hf.filename}")
            # Stack along new axis 0 -> (n_sources, channels, samples)
            return np.stack(arrays, axis=0).astype(np.float32)

        # Fallthrough
        raise KeyError("Index entry must contain either 'key' or 'sources' for target lookup.")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.indexes[idx]

        hdf5_path = entry["hdf5_path"]
        start = int(entry["start_sample"])
        end = int(entry["end_sample"])

        # open HDF5
        hf = self._open_h5(hdf5_path)

        # read mixture
        if self.mixture_key not in hf:
            raise KeyError(f"Mixture key '{self.mixture_key}' not present in {hdf5_path}")

        mix_np = self._read_dataset_segment(hf, self.mixture_key, start, end)  # (C, L)

        # read target(s)
        target_np = self._get_target_arrays(hf, entry)  # (C, L) or (n_targets, C, L)

        # Convert to torch tensors
        mix_t = torch.from_numpy(mix_np).float()  # (C, L)
        # If multiple targets were stacked, return as tensor with shape (n_targets, C, L)
        target_t = torch.from_numpy(target_np).float()

        # Optionally convert to mono by averaging channels
        if self.to_mono:
            mix_t = mix_t.mean(dim=0, keepdim=True)  # (1, L)
            # if target_t has shape (n_targets, C, L) -> average channels axis 1
            if target_t.ndim == 3:
                # (n_targets, C, L) -> (n_targets, L) -> add channel dim 1
                target_t = target_t.mean(dim=1, keepdim=True)  # (n_targets, 1, L)
            else:
                target_t = target_t.mean(dim=0, keepdim=True)  # (1, L)

        sample = {
            "mix": mix_t,          # (C, L) or (1, L) if to_mono
            "target": target_t,    # (C, L) or (n_targets, C, L) or (n_targets,1,L)
            "hdf5_path": hdf5_path,
            "start_sample": start,
            "end_sample": end,
        }
        if "source" in entry:
            sample["source"] = entry["source"]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def close(self):
        """Close all cached HDF5 file handles. Call this when you are done with dataset."""
        for p, f in list(self._h5_cache.items()):
            try:
                f.close()
            except Exception:
                pass
            self._h5_cache.pop(p, None)

    def __del__(self):
        # best-effort cleanup when dataset is garbage collected
        try:
            self.close()
        except Exception:
            pass


# ---------------------------
# Collate function & helpers
# ---------------------------

def default_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Batch collate: stack 'mix' and 'target' tensors across batch dimension.
    Assumes all segments have equal length (they do if produced by the indexer).
    Returns a dict:
      { "mix": (B, C, L), "target": (B, C, L) or (B, n_targets, C, L), "meta": {...} }
    """
    mixes = [b["mix"] for b in batch]
    targets = [b["target"] for b in batch]

    # If targets have dimension (n_targets, C, L) we want final shape (B, n_targets, C, L)
    if targets[0].ndim == 3:
        # stack into (B, n_targets, C, L)
        stacked_targets = torch.stack(targets, dim=0)
    else:
        # targets shape (C, L) -> stack to (B, C, L)
        stacked_targets = torch.stack(targets, dim=0)

    stacked_mixes = torch.stack(mixes, dim=0)  # (B, C, L)

    meta = {
        "hdf5_path": [b["hdf5_path"] for b in batch],
        "start_sample": [b["start_sample"] for b in batch],
        "end_sample": [b["end_sample"] for b in batch],
    }
    if "source" in batch[0]:
        meta["source"] = [b.get("source", None) for b in batch]

    return {"mix": stacked_mixes, "target": stacked_targets, "meta": meta}
