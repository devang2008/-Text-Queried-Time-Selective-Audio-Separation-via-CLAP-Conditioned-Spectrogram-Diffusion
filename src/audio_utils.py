"""
Audio processing utilities for loading, STFT, ISTFT, spectrograms, and time gating.
"""
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """
    Load audio file as mono float32.
    
    Args:
        path: Path to audio file
        sr: Target sample rate
        
    Returns:
        Audio waveform as float32 numpy array
    """
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)

def stft(y: np.ndarray, n_fft: int = 1024, hop: int = 256):
    """
    Compute Short-Time Fourier Transform.
    
    Args:
        y: Audio waveform
        n_fft: FFT window size
        hop: Hop length
        
    Returns:
        Tuple of (stft_complex, magnitude, phase)
    """
    stft_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop, window='hann')
    mag = np.abs(stft_complex)
    phase = np.angle(stft_complex)
    return stft_complex, mag, phase

def istft(mag: np.ndarray, phase: np.ndarray, hop: int = 256) -> np.ndarray:
    """
    Inverse STFT to reconstruct audio.
    
    Args:
        mag: Magnitude spectrogram
        phase: Phase spectrogram
        hop: Hop length
        
    Returns:
        Reconstructed audio waveform
    """
    stft_complex = mag * np.exp(1j * phase)
    y = librosa.istft(stft_complex, hop_length=hop, window='hann')
    return y

def log_mel(y: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:
    """
    Compute log-mel spectrogram.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        n_mels: Number of mel bands
        
    Returns:
        Log-mel spectrogram as float32
    """
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec.astype(np.float32)

def save_spectrogram_png(mag_or_mel: np.ndarray, out_path: str, title: str = "Spectrogram"):
    """
    Save spectrogram as PNG with colorbar and dB scaling.
    
    Args:
        mag_or_mel: Magnitude or mel spectrogram
        out_path: Output file path
        title: Plot title
    """
    # Convert to dB scale
    db = librosa.amplitude_to_db(mag_or_mel + 1e-10, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(db, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (bins)')
    plt.tight_layout()
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

def time_gate(n_frames: int, sr: int, hop: int, t0: float, t1: float, fade_ms: float = 70) -> np.ndarray:
    """
    Create a time gate with raised-cosine fades.
    
    Args:
        n_frames: Total number of time frames
        sr: Sample rate
        hop: Hop length
        t0: Start time in seconds
        t1: End time in seconds
        fade_ms: Fade duration in milliseconds
        
    Returns:
        1D gate array of shape [n_frames] with values in [0, 1]
    """
    gate = np.zeros(n_frames, dtype=np.float32)
    
    # Convert time to frame indices
    frame0 = int(t0 * sr / hop)
    frame1 = int(t1 * sr / hop)
    
    # Clamp to valid range
    frame0 = max(0, frame0)
    frame1 = min(n_frames, frame1)
    
    if frame1 <= frame0:
        return gate
    
    # Compute fade length in frames
    fade_frames = int(fade_ms * 0.001 * sr / hop)
    fade_frames = max(1, fade_frames)
    
    # Main gate region
    gate[frame0:frame1] = 1.0
    
    # Apply raised-cosine fade-in
    if fade_frames > 0 and frame0 + fade_frames < frame1:
        fade_in = np.linspace(0, np.pi/2, fade_frames)
        gate[frame0:frame0+fade_frames] = np.sin(fade_in) ** 2
    
    # Apply raised-cosine fade-out
    if fade_frames > 0 and frame1 - fade_frames > frame0:
        fade_out = np.linspace(np.pi/2, np.pi, fade_frames)
        gate[frame1-fade_frames:frame1] = np.sin(fade_out) ** 2
    
    return gate

def apply_time_gate_to_mask(mask: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """
    Apply time gate to a 2D mask by broadcasting.
    
    Args:
        mask: 2D mask of shape [freq_bins, time_frames]
        gate: 1D gate of shape [time_frames]
        
    Returns:
        Gated mask
    """
    return mask * gate[np.newaxis, :]
