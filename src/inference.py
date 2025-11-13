"""
Inference script for time-selective, text-guided audio separation.
Implements the complete pipeline with time-gated synthesis.
"""
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
from typing import List, Tuple, Literal

from model import TextConditionedUNet, TextAgnosticUNet
from audio_utils import load_audio, stft, istft, time_gate, apply_time_gate_to_mask
from clap_embed import text_embed
from config import SAMPLE_RATE, STFT_N_FFT, STFT_HOP


def load_model(checkpoint_path: str, baseline: bool = False, device='cpu'):
    """Load trained model from checkpoint."""
    if baseline:
        model = TextAgnosticUNet()
    else:
        model = TextConditionedUNet(clap_dim=1024)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model


def separate_with_unet(
    audio_path: str,
    prompt_text: str,
    time_windows: List[Tuple[float, float]],
    mode: Literal["keep", "remove"],
    model_path: str,
    output_path: str,
    baseline: bool = False,
    fade_ms: float = 70.0,
    sr: int = SAMPLE_RATE,
    n_fft: int = STFT_N_FFT,
    hop: int = STFT_HOP,
    device: str = 'cpu'
):
    """
    Time-selective, text-guided audio separation using trained UNet.
    
    Args:
        audio_path: Path to input audio file (Audio_In)
        prompt_text: Text description of target sound (Prompt_Text)
        time_windows: List of [start_sec, end_sec] pairs (Time_Windows)
        mode: "keep" or "remove" (Mode)
        model_path: Path to trained model checkpoint
        output_path: Path to save output audio (Audio_Out)
        baseline: Whether to use baseline model (text-agnostic)
        fade_ms: Fade duration in milliseconds for time windows
        sr: Sample rate
        n_fft: FFT size
        hop: Hop length
        device: 'cuda' or 'cpu'
    """
    device = torch.device(device)
    
    # 1. Load model
    print(f"\n{'='*60}")
    print("TIME-SELECTIVE TEXT-GUIDED AUDIO SEPARATION")
    print(f"{'='*60}")
    model = load_model(model_path, baseline=baseline, device=device)
    
    # 2. Load audio
    print(f"\nLoading audio: {audio_path}")
    audio_in = load_audio(audio_path, sr=sr)
    duration = len(audio_in) / sr
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sr} Hz")
    
    # 3. Get text embedding
    if not baseline:
        print(f"\nPrompt: '{prompt_text}'")
        text_emb = text_embed([prompt_text])
        text_emb_t = torch.from_numpy(text_emb).float().to(device)
    
    # 4. Compute spectrogram
    print("\nComputing STFT...")
    stft_complex, mix_mag, mix_phase = stft(audio_in, n_fft=n_fft, hop=hop)
    n_freq, n_frames = mix_mag.shape
    print(f"  Spectrogram shape: {mix_mag.shape}")
    
    # 5. Predict mask using UNet
    print("\nPredicting separation mask...")
    with torch.no_grad():
        mix_mag_t = torch.from_numpy(mix_mag).float().unsqueeze(0).unsqueeze(0).to(device)
        
        if baseline:
            pred_mask_t = model(mix_mag_t)
        else:
            pred_mask_t = model(mix_mag_t, text_emb_t)
        
        pred_mask = pred_mask_t.squeeze().cpu().numpy()
    
    print(f"  Predicted mask range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
    
    # 6. Create time gate
    print(f"\nApplying time windows: {time_windows}")
    print(f"  Mode: {mode}")
    print(f"  Fade duration: {fade_ms}ms")
    
    # Create combined time gate for all windows
    combined_gate = np.zeros(n_frames)
    for t0, t1 in time_windows:
        gate = time_gate(n_frames, sr, hop, t0, t1, fade_ms=fade_ms)
        combined_gate = np.maximum(combined_gate, gate)  # Combine multiple windows
    
    # 7. Apply time gate to mask
    final_mask = apply_time_gate_to_mask(pred_mask, combined_gate)
    
    # Count gated frames
    active_frames = np.sum(combined_gate > 0.01)
    print(f"  Active frames: {active_frames}/{n_frames} ({100*active_frames/n_frames:.1f}%)")
    
    # 8. Apply mask based on mode
    if mode == "keep":
        # Keep target sound
        separated_mag = mix_mag * final_mask
    else:  # mode == "remove"
        # Remove target sound (keep residual)
        separated_mag = mix_mag * (1 - final_mask)
    
    # 9. Reconstruct audio
    print("\nReconstructing audio with ISTFT...")
    audio_out = istft(separated_mag, mix_phase, hop=hop)
    
    # Ensure same length as input
    if len(audio_out) < len(audio_in):
        audio_out = np.pad(audio_out, (0, len(audio_in) - len(audio_out)))
    else:
        audio_out = audio_out[:len(audio_in)]
    
    # 10. Outside window: copy input audio (no change)
    print("Applying time-selective gating...")
    
    # Create sample-level gate
    sample_gate = np.zeros(len(audio_in))
    for t0, t1 in time_windows:
        start_sample = int(t0 * sr)
        end_sample = int(t1 * sr)
        
        # Apply fade in/out at sample level
        fade_samples = int(fade_ms * sr / 1000)
        
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        window_gate = np.ones(end_sample - start_sample)
        window_gate[:fade_samples] *= fade_in
        window_gate[-fade_samples:] *= fade_out
        
        sample_gate[start_sample:end_sample] = np.maximum(
            sample_gate[start_sample:end_sample],
            window_gate
        )
    
    # Blend separated audio inside windows, keep original outside
    audio_out = sample_gate * audio_out + (1 - sample_gate) * audio_in
    
    # 11. Save output
    print(f"\nSaving output: {output_path}")
    sf.write(output_path, audio_out, sr)
    print(f"  Length: {len(audio_out)/sr:.2f}s")
    
    # 12. Compute metrics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Input:  {audio_path}")
    print(f"Output: {output_path}")
    print(f"Prompt: {prompt_text}")
    print(f"Mode:   {mode}")
    print(f"Windows: {time_windows}")
    print(f"Model:  {'Baseline (no text)' if baseline else 'Text-conditioned'}")
    
    # Check outside-window preservation
    outside_mask = (sample_gate < 0.01)
    if np.any(outside_mask):
        outside_diff = np.abs(audio_out[outside_mask] - audio_in[outside_mask])
        max_diff = outside_diff.max()
        mean_diff = outside_diff.mean()
        print(f"\nOutside-window preservation:")
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        if max_diff < 1e-6:
            print(f"  ✓ Perfect preservation!")
    
    print(f"\n{'='*60}\n")
    
    return {
        'audio_out': audio_out,
        'audio_in': audio_in,
        'pred_mask': pred_mask,
        'final_mask': final_mask,
        'time_gate': sample_gate
    }


def main(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    
    # Parse time windows
    time_windows = []
    for i in range(0, len(args.time_windows), 2):
        t0 = args.time_windows[i]
        t1 = args.time_windows[i + 1]
        time_windows.append((t0, t1))
    
    # Run separation
    separate_with_unet(
        audio_path=args.audio_in,
        prompt_text=args.prompt_text,
        time_windows=time_windows,
        mode=args.mode,
        model_path=args.model_path,
        output_path=args.audio_out,
        baseline=args.baseline,
        fade_ms=args.fade_ms,
        sr=SAMPLE_RATE,
        n_fft=STFT_N_FFT,
        hop=STFT_HOP,
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time-selective, text-guided audio separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keep dog bark from 0-4s
  python inference.py --audio-in mix.wav --audio-out out.wav \\
      --prompt-text "dog bark" --mode keep --time-windows 0 4 \\
      --model-path checkpoints/text_conditioned_unet_best.pth
  
  # Remove speech from 2-6s and 8-10s
  python inference.py --audio-in mix.wav --audio-out out.wav \\
      --prompt-text "speech" --mode remove --time-windows 2 6 8 10 \\
      --model-path checkpoints/text_conditioned_unet_best.pth
        """
    )
    
    # Required arguments
    parser.add_argument('--audio-in', type=str, required=True,
                       help='Input audio file (Audio_In)')
    parser.add_argument('--audio-out', type=str, required=True,
                       help='Output audio file (Audio_Out)')
    parser.add_argument('--prompt-text', type=str, required=True,
                       help='Text description of target sound (Prompt_Text)')
    parser.add_argument('--time-windows', type=float, nargs='+', required=True,
                       help='Time windows as pairs: t0_1 t1_1 t0_2 t1_2 ... (in seconds)')
    parser.add_argument('--mode', type=str, choices=['keep', 'remove'], required=True,
                       help='Keep or remove target sound (Mode)')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Optional arguments
    parser.add_argument('--baseline', action='store_true',
                       help='Use baseline model (text-agnostic)')
    parser.add_argument('--fade-ms', type=float, default=70.0,
                       help='Fade duration in milliseconds (default: 70)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU inference')
    
    args = parser.parse_args()
    
    # Validate time windows
    if len(args.time_windows) % 2 != 0:
        parser.error("--time-windows must have even number of values (pairs of start/end times)")
    
    main(args)
