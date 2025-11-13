"""
CLAP embeddings for text and audio using laion-clap.
Provides caching and similarity computation.
"""
import numpy as np
import torch
from typing import List, Dict
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Global model cache
_clap_model = None
_text_cache: Dict[str, np.ndarray] = {}

def load_clap():
    """
    Load CLAP model once and cache globally.
    Downloads default HuggingFace weights on first call.
    
    Returns:
        CLAP model instance
    """
    global _clap_model
    
    if _clap_model is not None:
        return _clap_model
    
    try:
        from msclap import CLAP
        
        # Initialize CLAP with fusion disabled
        print("Loading CLAP model...")
        _clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
        print(f"CLAP model loaded (device: {'CUDA' if torch.cuda.is_available() else 'CPU'})")
        
    except ImportError:
        print("Error: msclap not found. Installing laion-clap...")
        raise ImportError("Please install: pip install laion-clap")
    
    return _clap_model

def text_embed(texts: List[str]) -> np.ndarray:
    """
    Get text embeddings with caching.
    
    Args:
        texts: List of text prompts
        
    Returns:
        Numpy array of shape [N, D] with text embeddings
    """
    model = load_clap()
    embeddings = []
    
    for text in texts:
        # Check cache
        if text in _text_cache:
            embeddings.append(_text_cache[text])
        else:
            # Compute embedding
            with torch.no_grad():
                emb = model.get_text_embeddings([text])
                emb_np = emb.cpu().numpy()[0]
                _text_cache[text] = emb_np
                embeddings.append(emb_np)
    
    return np.array(embeddings)

def audio_embed(waveforms: List[np.ndarray], srs: List[int]) -> np.ndarray:
    """
    Get audio embeddings. Saves to temp files since msclap expects file paths.
    
    Args:
        waveforms: List of audio waveforms
        srs: List of sample rates corresponding to each waveform
        
    Returns:
        Numpy array of shape [N, D] with audio embeddings
    """
    import soundfile as sf
    import tempfile
    import os
    
    model = load_clap()
    embeddings = []
    
    # msclap expects file paths, so we need to save to temp files
    temp_files = []
    
    try:
        for i, (wav, sr) in enumerate(zip(waveforms, srs)):
            # Create temp file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            temp_files.append(temp_path)
            
            # Save audio to temp file at 48kHz (CLAP's expected rate)
            target_sr = 48000
            if sr != target_sr:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            
            sf.write(temp_path, wav, target_sr)
        
        # Get embeddings from files
        with torch.no_grad():
            emb = model.get_audio_embeddings(temp_files, resample=False)
            embeddings = emb.cpu().numpy()
    
    finally:
        # Clean up temp files
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
    
    return embeddings

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between two sets of embeddings.
    
    Args:
        a: Embeddings of shape [N, D]
        b: Embeddings of shape [M, D]
        
    Returns:
        Similarity matrix of shape [N, M]
    """
    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    
    # Compute cosine similarity
    sim = np.dot(a_norm, b_norm.T)
    return sim
