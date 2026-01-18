#!/usr/bin/env python3
"""
End-to-end inference script for Splitify music separation.

This script takes any audio file and separates it into 4 stems:
- vocals
- bass  
- drums
- other (melody/instruments)

Usage:
    python inference.py --model_path checkpoints/model.pt --input song.mp3 --output_dir separated/
"""

import os
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Import your model and STFT utilities
from learning.multistem.model import UNetMulti
from learning.multistem.stft_utils import stft_waveform, mag_and_phase_from_complex, istft_from_mag_phase


def load_audio_file(filepath: str, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
    """
    Load any audio file and convert to target sample rate.
    
    Returns:
        waveform: (channels, samples) tensor
        sample_rate: int
    """
    print(f"Loading audio from: {filepath}")
    
    # Load audio file
    waveform, orig_sr = torchaudio.load(filepath)
    
    # Resample if needed
    if orig_sr != target_sr:
        print(f"Resampling from {orig_sr} Hz to {target_sr} Hz")
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)
    
    print(f"Audio loaded: {waveform.shape[0]} channels, {waveform.shape[1]} samples, {target_sr} Hz")
    return waveform, target_sr


def separate_stems(
    model: UNetMulti,
    waveform: torch.Tensor,
    stems: list = ["vocals", "bass", "drums", "other"],
    device: str = "cpu",
    n_fft: int = 2048,
    hop_length: int = 512,
    chunk_length: int = 44100 * 10  # 10 seconds chunks to avoid memory issues
) -> Dict[str, torch.Tensor]:
    """
    Separate audio into stems using the trained model.
    
    Args:
        model: Trained UNetMulti model
        waveform: Input audio (channels, samples)
        stems: List of stem names
        device: Device to run inference on
        n_fft: FFT size for STFT
        hop_length: Hop length for STFT
        chunk_length: Process audio in chunks to save memory
        
    Returns:
        Dictionary of separated stems {stem_name: waveform_tensor}
    """
    model.eval()
    
    # Convert to mono by averaging channels
    if waveform.shape[0] > 1:
        mono_waveform = torch.mean(waveform, dim=0, keepdim=True)  # (1, samples)
    else:
        mono_waveform = waveform
    
    total_samples = mono_waveform.shape[1]
    separated_stems = {stem: [] for stem in stems}
    
    print(f"Processing audio in chunks of {chunk_length/44100:.1f} seconds...")
    
    with torch.no_grad():
        for start_idx in range(0, total_samples, chunk_length):
            end_idx = min(start_idx + chunk_length, total_samples)
            chunk = mono_waveform[:, start_idx:end_idx].to(device)  # (1, chunk_samples)
            
            # Add batch dimension and compute STFT
            chunk_batch = chunk.unsqueeze(0)  # (1, 1, chunk_samples)
            
            # Compute STFT
            complex_spec = stft_waveform(
                chunk_batch, 
                n_fft=n_fft, 
                hop_length=hop_length
            )  # (1, 1, freq_bins, time_frames)
            
            # Get magnitude and phase
            magnitude, phase = mag_and_phase_from_complex(complex_spec)
            
            # Model expects (batch, 1, freq, time)
            magnitude_input = magnitude.squeeze(1)  # (1, 1, freq, time)
            
            # Run model to get masks
            stem_masks = model(magnitude_input)  # (1, n_stems, freq, time)
            
            # Apply masks to get separated magnitudes
            for i, stem_name in enumerate(stems):
                mask = stem_masks[0, i:i+1]  # (1, freq, time)
                separated_mag = magnitude.squeeze(1) * mask  # (1, freq, time)
                
                # Reconstruct waveform using original phase
                separated_complex = separated_mag.unsqueeze(1) * phase  # (1, 1, freq, time)
                
                separated_wave = istft_from_mag_phase(
                    separated_mag.unsqueeze(1),  # Add channel dim back
                    phase,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    length=chunk.shape[-1]
                )  # (1, 1, chunk_samples)
                
                separated_stems[stem_name].append(separated_wave.squeeze().cpu())
    
    # Concatenate chunks for each stem
    final_stems = {}
    for stem_name in stems:
        final_stems[stem_name] = torch.cat(separated_stems[stem_name], dim=0)
        
        # If original was stereo, duplicate mono to stereo
        if waveform.shape[0] > 1:
            final_stems[stem_name] = final_stems[stem_name].unsqueeze(0).repeat(waveform.shape[0], 1)
    
    return final_stems


def save_separated_stems(
    separated_stems: Dict[str, torch.Tensor],
    output_dir: str,
    sample_rate: int = 44100,
    input_filename: str = "audio"
) -> None:
    """Save separated stems as audio files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(input_filename).stem
    
    print(f"Saving separated stems to: {output_dir}")
    
    for stem_name, waveform in separated_stems.items():
        # Ensure waveform is 2D (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        output_file = output_path / f"{base_name}_{stem_name}.wav"
        torchaudio.save(str(output_file), waveform, sample_rate)
        print(f"✅ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Separate music into vocals, bass, drums, and other stems"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to input audio file (mp3, wav, flac, etc.)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="separated",
        help="Directory to save separated stems (default: separated/)"
    )
    parser.add_argument(
        "--stems", 
        nargs="+", 
        default=["vocals", "bass", "drums", "other"],
        help="Stem names to separate (default: vocals bass drums other)"
    )
    parser.add_argument(
        "--sample_rate", 
        type=int, 
        default=44100,
        help="Target sample rate (default: 44100)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        return
    
    try:
        # Load model
        print(f"Loading model from: {args.model_path}")
        model = UNetMulti(in_channels=1, n_outputs=len(args.stems))
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
        
        # Load audio
        waveform, sample_rate = load_audio_file(args.input, args.sample_rate)
        
        # Separate stems
        print("🎵 Separating stems...")
        separated_stems = separate_stems(
            model=model,
            waveform=waveform,
            stems=args.stems,
            device=device
        )
        
        # Save results
        save_separated_stems(
            separated_stems=separated_stems,
            output_dir=args.output_dir,
            sample_rate=sample_rate,
            input_filename=args.input
        )
        
        print("🎉 Separation complete!")
        print(f"Check the '{args.output_dir}' directory for your separated stems.")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
