import torch
import math
from typing import Optional, Tuple

# Default STFT params (good baseline for music)
DEFAULT_N_FFT = 2048
DEFAULT_HOP = 512
DEFAULT_WIN_LENGTH = 2048
DEFAULT_WINDOW = "hann"

_eps = 1e-8


def _get_window(window_name: str, win_length: int, device: torch.device, dtype: torch.dtype):
    if window_name == "hann":
        return torch.hann_window(win_length, device=device, dtype=dtype)
    raise ValueError("Only 'hann' supported by _get_window helper. You can add more if needed.")


# -----------------------------------------------------------------------------
# Core functional helpers (as you already wrote)
# -----------------------------------------------------------------------------

def stft_waveform(
    wave: torch.Tensor,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP,
    win_length: int = None,  # Changed to None so we can auto-set it
    window_name: str = DEFAULT_WINDOW,
    center: bool = True,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """Compute STFT for waveforms."""
    # Auto-set win_length to n_fft if not provided or if it exceeds n_fft
    if win_length is None or win_length > n_fft:
        win_length = n_fft
    
    if wave.ndim == 2:  # (B, T)
        wave = wave.unsqueeze(1)
    B, C, T = wave.shape
    device, dtype = wave.device, wave.dtype
    window = _get_window(window_name, win_length, device, dtype)
    flat = wave.reshape(B * C, T)
    complex_spec = torch.stft(
        flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=False,
        return_complex=True,
    )
    F, T_frames = complex_spec.shape[1], complex_spec.shape[2]
    return complex_spec.reshape(B, C, F, T_frames)


def mag_and_phase_from_complex(complex_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split complex spectrogram into magnitude and phase."""
    mag = complex_spec.abs()
    phase = complex_spec / (mag + _eps)
    return mag, phase


def istft_from_complex(
    complex_spec: torch.Tensor,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP,
    win_length: int = None,  # Changed to None
    window_name: str = DEFAULT_WINDOW,
    center: bool = True,
    length: Optional[int] = None,
) -> torch.Tensor:
    """Inverse STFT to waveform."""
    # Auto-set win_length to n_fft if not provided or if it exceeds n_fft
    if win_length is None or win_length > n_fft:
        win_length = n_fft
        
    B, C, F, T_frames = complex_spec.shape
    device, dtype = complex_spec.device, complex_spec.real.dtype
    window = _get_window(window_name, win_length, device, dtype)
    flat = complex_spec.reshape(B * C, F, T_frames)
    wave_flat = torch.istft(
        flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=False,
        length=length,
    )
    samples = wave_flat.shape[1]
    return wave_flat.reshape(B, C, samples)


def magphase_from_wave(
    wave: torch.Tensor,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP,
    win_length: int = None,  # Changed to None
    window_name: str = DEFAULT_WINDOW,
    center: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper to get magnitude and phase from waveform."""
    complex_spec = stft_waveform(
        wave, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window_name=window_name, center=center
    )
    mag, phase = mag_and_phase_from_complex(complex_spec)
    return mag, phase


def istft_from_mag_phase(
    mag: torch.Tensor,
    phase: torch.Tensor,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP,
    win_length: int = None,  # Changed to None
    window_name: str = DEFAULT_WINDOW,
    center: bool = True,
    length: Optional[int] = None,
) -> torch.Tensor:
    """Reconstruct waveform from magnitude and phase."""
    complex_spec = mag * phase
    return istft_from_complex(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_name=window_name,
        center=center,
        length=length,
    )


# -----------------------------------------------------------------------------
# Class-based wrappers (for easy plug-and-play use in model or dataloader)
# -----------------------------------------------------------------------------

class stft(torch.nn.Module):
    """Wrapper module for computing STFT magnitude and phase."""
    def __init__(
        self,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP,
        win_length: int = None,  # Changed to None
        window_name: str = DEFAULT_WINDOW,
        center: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Auto-set win_length to n_fft if not provided or if it exceeds n_fft
        if win_length is None or win_length > n_fft:
            win_length = n_fft
        self.win_length = win_length
        self.window_name = window_name
        self.center = center

    def forward(self, wave: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag, phase = magphase_from_wave(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_name=self.window_name,
            center=self.center,
        )
        return mag, phase


class istft(torch.nn.Module):
    """Wrapper module for waveform reconstruction from magnitude and phase."""
    def __init__(
        self,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP,
        win_length: int = None,  # Changed to None
        window_name: str = DEFAULT_WINDOW,
        center: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Auto-set win_length to n_fft if not provided or if it exceeds n_fft
        if win_length is None or win_length > n_fft:
            win_length = n_fft
        self.win_length = win_length
        self.window_name = window_name
        self.center = center

    def forward(
        self,
        mag: torch.Tensor,
        phase: torch.Tensor,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        return istft_from_mag_phase(
            mag,
            phase,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_name=self.window_name,
            center=self.center,
            length=length,
        )
