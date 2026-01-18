import os
import torch
import torchaudio
import numpy as np
from model import UNetMulti  # Import your updated model
from tqdm import tqdm


def load_audio(filepath, sr=44100):
    """Load an audio file and convert to mono."""
    waveform, file_sr = torchaudio.load(filepath)
    if file_sr != sr:
        waveform = torchaudio.functional.resample(waveform, file_sr, sr)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # mono
    return waveform, sr


def stft_mag_phase(waveform, n_fft=1024, hop_length=512):
    """Compute magnitude and phase of STFT."""
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    return mag, phase


def istft_reconstruct(mag, phase, hop_length=512):
    """Reconstruct waveform from magnitude and phase."""
    complex_stft = mag * torch.exp(1j * phase)
    waveform = torch.istft(complex_stft, hop_length=hop_length)
    return waveform


def separate_stems(model, mix_mag, mix_phase, stems, device="cuda"):
    """Apply the model to separate all stems."""
    model.eval()
    with torch.no_grad():
        x = mix_mag.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, F, T)
        outputs = model(x)  # (1, num_stems, F, T)

        separated_waveforms = {}
        for i, stem_name in enumerate(stems):
            mask = outputs[0, i].cpu()
            masked_mag = mix_mag * mask
            waveform = istft_reconstruct(masked_mag, mix_phase)
            separated_waveforms[stem_name] = waveform
    return separated_waveforms


def save_waveforms(stem_waveforms, output_dir, sr=44100):
    os.makedirs(output_dir, exist_ok=True)
    for stem_name, waveform in stem_waveforms.items():
        out_path = os.path.join(output_dir, f"{stem_name}.wav")
        torchaudio.save(out_path, waveform.unsqueeze(0), sr)
        print(f"✅ Saved {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate multi-output U-Net for stem separation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to input mix audio")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save separated stems")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Target sampling rate")
    parser.add_argument("--stems", nargs="+", default=["vocals", "drums", "bass", "other"], help="Stem names")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_path}...")
    model = UNetMulti(in_channels=1, n_outputs=len(args.stems)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    print(f"Loading audio from {args.input_audio}...")
    waveform, sr = load_audio(args.input_audio, sr=args.sample_rate)

    print("Computing STFT...")
    mix_mag, mix_phase = stft_mag_phase(waveform)

    print("Separating stems...")
    separated_waveforms = separate_stems(model, mix_mag, mix_phase, args.stems, device=device)

    print("Saving outputs...")
    save_waveforms(separated_waveforms, args.output_dir, sr=args.sample_rate)

    print("✅ Done! All stems saved in:", args.output_dir)
