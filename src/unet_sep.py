"""
UNet-based separation for web interface.
Wrapper around inference.py for use in FastAPI server.
"""
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import uuid
from typing import Literal, Optional

from model import TextConditionedUNet
from audio_utils import load_audio, stft, istft, time_gate, apply_time_gate_to_mask, save_spectrogram_png
from clap_embed import text_embed
from config import SAMPLE_RATE, STFT_N_FFT, STFT_HOP, IMG_DIR, AUDIO_DIR

# Global model cache
_unet_model = None
_device = None

def load_unet_model(model_path: str = None):
    """Load and cache the trained UNet model."""
    global _unet_model, _device
    
    if _unet_model is not None:
        return _unet_model, _device
    
    # Default model path
    if model_path is None:
        model_path = Path(__file__).parent.parent / "checkpoints" / "text_conditioned_unet_best.pth"
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Determine device
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    _unet_model = TextConditionedUNet(clap_dim=1024)
    checkpoint = torch.load(model_path, map_location=_device)
    _unet_model.load_state_dict(checkpoint['model_state_dict'])
    _unet_model = _unet_model.to(_device)
    _unet_model.eval()
    
    print(f"✓ UNet model loaded from {model_path}")
    print(f"  Device: {_device}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return _unet_model, _device


def separate_with_unet(
    audio_path: str,
    prompt: str,
    mode: Literal["keep", "remove"] = "keep",
    t0: float = 0.0,
    t1: Optional[float] = None,
    fade_ms: float = 70.0,
    sr: int = SAMPLE_RATE,
    n_fft: int = STFT_N_FFT,
    hop: int = STFT_HOP,
    model_path: str = None
) -> dict:
    """
    Separate audio using trained UNet model with time-selective synthesis.
    
    Args:
        audio_path: Path to input audio file
        prompt: Text description of sound to keep/remove
        mode: "keep" or "remove"
        t0: Start time in seconds
        t1: End time in seconds (None = full duration)
        fade_ms: Fade duration in milliseconds
        sr: Sample rate
        n_fft: FFT size
        hop: Hop length
        model_path: Path to trained model (None = use default)
        
    Returns:
        Dictionary with file paths and confidence score
    """
    sep_id = str(uuid.uuid4())[:8]
    
    # Load model
    model, device = load_unet_model(model_path)
    
    # Load audio
    audio_in = load_audio(audio_path, sr=sr)
    duration = len(audio_in) / sr
    
    if t1 is None:
        t1 = duration
    
    # Get text embedding [1, 1024]
    text_emb = text_embed([prompt])
    text_emb_t = torch.from_numpy(text_emb).float().to(device)
    
    # Compute spectrogram
    stft_complex, mix_mag, mix_phase = stft(audio_in, n_fft=n_fft, hop=hop)
    n_freq, n_frames = mix_mag.shape
    
    # Predict mask using UNet
    with torch.no_grad():
        # Shape: [batch_size=1, channels=1, freq_bins, time_frames]
        mix_mag_t = torch.from_numpy(mix_mag).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Ensure text embedding has correct shape [batch_size=1, embedding_dim]
        if text_emb_t.dim() == 3:  # If shape is [1, 1, 1024]
            text_emb_t = text_emb_t.squeeze(1)  # Convert to [1, 1024]
        
        # Print shapes for debugging
        print(f"Magnitude shape: {mix_mag_t.shape}")
        print(f"Text embedding shape: {text_emb_t.shape}")
        
        pred_mask_t = model(mix_mag_t, text_emb_t)
        pred_mask = pred_mask_t.squeeze().cpu().numpy()
    
    # Create time gate
    gate = time_gate(n_frames, sr, hop, t0, t1, fade_ms=fade_ms)
    
    # Normalize and enhance mask
    pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
    pred_mask = pred_mask ** 1.5  # Enhance separation (increase contrast)
    
    # Apply time gate to mask
    final_mask = apply_time_gate_to_mask(pred_mask, gate)
    
    # Apply mask based on mode with smoothing
    eps = 1e-8  # prevent division by zero
    mix_mag_smoothed = mix_mag + eps
    
    if mode == "keep":
        separated_mag = mix_mag_smoothed * final_mask
        residual_mag = mix_mag_smoothed * (1 - final_mask)
    else:  # remove
        separated_mag = mix_mag_smoothed * (1 - final_mask)
        residual_mag = mix_mag_smoothed * final_mask
        
    # Remove epsilon
    separated_mag = np.maximum(separated_mag - eps, 0)
    residual_mag = np.maximum(residual_mag - eps, 0)
    
    # Reconstruct audio
    audio_out = istft(separated_mag, mix_phase, hop=hop)
    residual_audio = istft(residual_mag, mix_phase, hop=hop)
    
    # Ensure same length as input
    if len(audio_out) < len(audio_in):
        audio_out = np.pad(audio_out, (0, len(audio_in) - len(audio_out)))
    else:
        audio_out = audio_out[:len(audio_in)]
    
    if len(residual_audio) < len(audio_in):
        residual_audio = np.pad(residual_audio, (0, len(audio_in) - len(residual_audio)))
    else:
        residual_audio = residual_audio[:len(audio_in)]
        
    # Normalize audio (make it audible)
    def normalize_audio(audio):
        max_val = np.abs(audio).max()
        if max_val > 1e-8:
            return audio * (0.95 / max_val)
        return audio
    
    audio_out = normalize_audio(audio_out)
    residual_audio = normalize_audio(residual_audio)
    
    # Apply time-selective gating at sample level
    sample_gate = np.zeros(len(audio_in))
    start_sample = int(t0 * sr)
    end_sample = int(t1 * sr)
    fade_samples = int(fade_ms * sr / 1000)
    
    # Create smooth fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    window_gate = np.ones(end_sample - start_sample)
    if len(window_gate) > 2 * fade_samples:
        window_gate[:fade_samples] *= fade_in
        window_gate[-fade_samples:] *= fade_out
    
    sample_gate[start_sample:end_sample] = window_gate
    
    # Blend: processed inside windows, original outside
    audio_out = sample_gate * audio_out + (1 - sample_gate) * audio_in
    
    # Save output files
    out_wav_path = str(Path(AUDIO_DIR) / f"{sep_id}_output.wav")
    residual_wav_path = str(Path(AUDIO_DIR) / f"{sep_id}_residual.wav")
    
    sf.write(out_wav_path, audio_out, sr)
    sf.write(residual_wav_path, residual_audio, sr)
    
    # Save spectrograms
    mix_spec_path = str(Path(IMG_DIR) / f"{sep_id}_mix.png")
    out_spec_path = str(Path(IMG_DIR) / f"{sep_id}_output.png")
    mask_path = str(Path(IMG_DIR) / f"{sep_id}_mask.png")
    
    save_spectrogram_png(mix_mag, mix_spec_path, "Input Mixture")
    save_spectrogram_png(separated_mag, out_spec_path, f"Output ({mode})")
    
    # Save mask visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.imshow(final_mask, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Mask strength')
    plt.title(f'Separation Mask ({mode}) - UNet')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (bins)')
    plt.tight_layout()
    plt.savefig(mask_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Compute confidence (simplified - compare energy inside window)
    window_mix = audio_in[start_sample:end_sample]
    window_out = audio_out[start_sample:end_sample]
    
    if len(window_mix) > 0 and len(window_out) > 0:
        energy_in = np.sum(window_mix ** 2)
        energy_out = np.sum(window_out ** 2)
        confidence = float(energy_out / (energy_in + 1e-8))
    else:
        confidence = 0.0
    
    return {
        "out_wav": out_wav_path,
        "residual_wav": residual_wav_path,
        "mask_png": mask_path,
        "mix_spec_png": mix_spec_path,
        "out_spec_png": out_spec_path,
        "confidence": confidence
    }


if __name__ == "__main__":
    # Test
    print("Testing UNet separation...")
    test_audio = "../data/ESC-50-master/ESC-50-master/audio/1-100032-A-0.wav"
    result = separate_with_unet(
        audio_path=test_audio,
        prompt="dog",
        mode="keep",
        t0=0.0,
        t1=5.0
    )
    print(f"✓ Separation complete!")
    print(f"  Output: {result['out_wav']}")
    print(f"  Confidence: {result['confidence']:.3f}")
