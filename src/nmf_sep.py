"""
NMF-based audio separation with CLAP text-querying.
"""
import numpy as np
import soundfile as sf
from sklearn.decomposition import NMF
from pathlib import Path
from typing import Optional, Literal
import uuid

from audio_utils import load_audio, stft, istft, save_spectrogram_png, time_gate, apply_time_gate_to_mask
from clap_embed import text_embed, audio_embed, cosine_sim
from config import SAMPLE_RATE, STFT_N_FFT, STFT_HOP, IMG_DIR, AUDIO_DIR

def separate_with_text(
    audio_path: str,
    prompt: str,
    mode: Literal["keep", "remove"] = "keep",
    k_components: int = 10,
    t0: float = 0.0,
    t1: Optional[float] = None,
    sr: int = SAMPLE_RATE,
    n_fft: int = STFT_N_FFT,
    hop: int = STFT_HOP
) -> dict:
    """
    Separate audio using NMF and CLAP text-querying.
    
    Args:
        audio_path: Path to input audio file
        prompt: Text description of sound to keep/remove
        mode: "keep" to isolate matching sounds, "remove" to suppress them
        k_components: Number of NMF components
        t0: Start time in seconds for processing window
        t1: End time in seconds (None = full duration)
        sr: Sample rate
        n_fft: FFT size
        hop: Hop length
        
    Returns:
        Dictionary with file paths and confidence score
    """

    sep_id = str(uuid.uuid4())[:8]
    
  
    y = load_audio(audio_path, sr=sr)
    duration = len(y) / sr
    
    if t1 is None:
        t1 = duration
    
    _, mag, phase = stft(y, n_fft=n_fft, hop=hop)
    n_freq_bins, n_frames = mag.shape
    

    nmf_model = NMF(
        n_components=k_components,
        init='nndsvda',
        beta_loss='kullback-leibler',
        solver='mu',
        max_iter=300,
        random_state=42
    )
    
    W = nmf_model.fit_transform(mag) 
    H = nmf_model.components_          
    
    component_audios = []
    component_masks = []
    
    for k in range(k_components):
      
        comp_mag = np.outer(W[:, k], H[k, :])
        
     
        sum_comp_mag = np.sum([np.outer(W[:, i], H[i, :]) for i in range(k_components)], axis=0) + 1e-8
        mask_k = comp_mag / sum_comp_mag
        component_masks.append(mask_k)
        
       
        comp_stft_mag = mask_k * mag
        audio_k = istft(comp_stft_mag, phase, hop=hop)
        component_audios.append(audio_k)
    
   
    prompt_emb = text_embed([prompt]) 
    
    
    comp_embs = audio_embed(component_audios, [sr] * k_components) 
    
   
    similarities = cosine_sim(comp_embs, prompt_emb).flatten() 
    
   
    sim_min = similarities.min()
    sim_max = similarities.max()
    if sim_max > sim_min:
        sim_normalized = (similarities - sim_min) / (sim_max - sim_min)
    else:
        sim_normalized = np.ones_like(similarities) / k_components
    
   
    combined_mask = np.zeros_like(mag)
    for k in range(k_components):
        combined_mask += component_masks[k] * sim_normalized[k]
    
   
    combined_mask = combined_mask / (combined_mask.max() + 1e-8)
    
 
    if t1 is not None and t1 > t0:
        gate = time_gate(n_frames, sr, hop, t0, t1, fade_ms=70)
        combined_mask = apply_time_gate_to_mask(combined_mask, gate)
    
  
    if mode == "keep":
        out_mag = mag * combined_mask
        residual_mag = mag * (1 - combined_mask)
    else: 
        out_mag = mag * (1 - combined_mask)
        residual_mag = mag * combined_mask
    
   
    out_audio = istft(out_mag, phase, hop=hop)
    residual_audio = istft(residual_mag, phase, hop=hop)
    
    
    out_wav_path = str(Path(AUDIO_DIR) / f"{sep_id}_output.wav")
    residual_wav_path = str(Path(AUDIO_DIR) / f"{sep_id}_residual.wav")
    
    sf.write(out_wav_path, out_audio, sr)
    sf.write(residual_wav_path, residual_audio, sr)
    
  
    mix_spec_path = str(Path(IMG_DIR) / f"{sep_id}_mix.png")
    out_spec_path = str(Path(IMG_DIR) / f"{sep_id}_output.png")
    mask_path = str(Path(IMG_DIR) / f"{sep_id}_mask.png")
    
    save_spectrogram_png(mag, mix_spec_path, "Input Mixture")
    save_spectrogram_png(out_mag, out_spec_path, f"Output ({mode})")
    
  
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.imshow(combined_mask, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Mask strength')
    plt.title(f'Separation Mask ({mode})')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (bins)')
    plt.tight_layout()
    plt.savefig(mask_path, dpi=100, bbox_inches='tight')
    plt.close()
    
  
    sample_start = int(t0 * sr)
    sample_end = int(t1 * sr) if t1 is not None else len(y)
    sample_start = max(0, sample_start)
    sample_end = min(len(y), sample_end)
    
    window_mix = y[sample_start:sample_end]
    window_out = out_audio[sample_start:min(sample_end, len(out_audio))]
    
    # Get embeddings for window
    if len(window_mix) > sr * 0.1 and len(window_out) > sr * 0.1: 
        mix_emb = audio_embed([window_mix], [sr])
        out_emb = audio_embed([window_out], [sr])
        
        mix_sim = cosine_sim(mix_emb, prompt_emb)[0, 0]
        out_sim = cosine_sim(out_emb, prompt_emb)[0, 0]
        
        confidence = float(out_sim - mix_sim)
    else:
        confidence = 0.0
    
    # 13. Return results
    return {
        "out_wav": out_wav_path,
        "residual_wav": residual_wav_path,
        "mask_png": mask_path,
        "mix_spec_png": mix_spec_path,
        "out_spec_png": out_spec_path,
        "confidence": confidence
    }
