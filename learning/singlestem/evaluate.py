# evaluate.py
import os
import torch
import soundfile as sf
import h5py
import numpy as np
from tqdm import tqdm

from model import UNet
from stft_utils import (
    stft_waveform,
    mag_and_phase_from_complex,
    istft_from_mag_phase,
)


@torch.no_grad()
def evaluate(
    model_ckpt: str,
    input_hdf5_dir: str,
    output_dir: str,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Evaluate trained U-Net on test MUSDB18 HDF5 files.
    It loads each mixture, performs STFT → mask prediction → ISTFT,
    and writes the separated stem to WAV files.
    """

    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading model checkpoint: {model_ckpt}")

    # Load trained model
    model = UNet(in_channels=1).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    # Iterate over test tracks
    h5_files = sorted([f for f in os.listdir(input_hdf5_dir) if f.endswith(".h5")])
    print(f"Found {len(h5_files)} test files in {input_hdf5_dir}")

    for filename in tqdm(h5_files, desc="Evaluating"):
        h5_path = os.path.join(input_hdf5_dir, filename)

        # --- 1. Load mixture ---
        with h5py.File(h5_path, "r") as hf:
            if "mixture" not in hf:
                print(f"⚠️ Missing 'mixture' key in {filename}, skipping.")
                continue
            mixture = hf["mixture"][:]  # shape (C, T)
        if mixture.ndim == 1:
            mixture = np.expand_dims(mixture, 0)

        mixture = torch.from_numpy(mixture).float().unsqueeze(0).to(device)  # (1, C, T)

        # --- 2. Compute STFT ---
        complex_spec = stft_waveform(
            mixture, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        mag, phase = mag_and_phase_from_complex(complex_spec)

        # --- 3. Predict mask & reconstruct ---
        mixture_mag = mag.unsqueeze(1)  # (B, 1, C, F, T) → treat channels separately
        B, C, F, T = mag.shape

        separated_mag = torch.zeros_like(mag)

        for c in range(C):
            # Pass channel through model
            input_mag = mag[:, c, :, :].unsqueeze(1)  # (B, 1, F, T)
            pred_mask = model(input_mag)
            separated_mag[:, c, :, :] = (pred_mask * input_mag).squeeze(1)

        # --- 4. Reconstruct waveform ---
        pred_wave = istft_from_mag_phase(
            separated_mag, phase, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )  # (B, C, T)

        # --- 5. Save separated stem ---
        pred_wave = pred_wave.squeeze(0).cpu().numpy().T  # (T, C)
        output_path = os.path.join(output_dir, filename.replace(".h5", "_vocals.wav"))
        sf.write(output_path, pred_wave, 44100)
        print(f"✅ Saved separated vocals: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input_hdf5_dir", type=str, required=True, help="Path to MUSDB18 test hdf5 directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save separated WAV files")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--win_length", type=int, default=2048)
    args = parser.parse_args()

    evaluate(
        model_ckpt=args.model_ckpt,
        input_hdf5_dir=args.input_hdf5_dir,
        output_dir=args.output_dir,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )
