"""
ESC-50 Dataset for on-the-fly synthetic mixture generation.
Creates training samples by mixing two clips at random SNR.
"""
import torch
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, List
import random

from audio_utils import load_audio, stft
from clap_embed import text_embed
from model import compute_irm
from config import SAMPLE_RATE, STFT_N_FFT, STFT_HOP


class ESC50MixtureDataset(Dataset):
    """
    ESC-50 dataset with on-the-fly mixture generation.
    
    For each sample:
    1. Randomly select two clips (target and interferer)
    2. Mix them at random SNR (-5dB to +5dB)
    3. Compute Ideal Ratio Mask (IRM) as ground truth
    4. Return mixture magnitude, text embedding, and IRM
    """
    
    def __init__(
        self,
        esc50_path: str,
        split: str = "train",
        train_fold_ids: List[int] = [1, 2, 3, 4],
        val_fold_ids: List[int] = [5],
        sr: int = SAMPLE_RATE,
        n_fft: int = STFT_N_FFT,
        hop: int = STFT_HOP,
        snr_range: Tuple[float, float] = (-5.0, 5.0),
        cache_embeddings: bool = True
    ):
        """
        Args:
            esc50_path: Path to ESC-50-master directory
            split: "train" or "val"
            train_fold_ids: Fold IDs for training (1-5)
            val_fold_ids: Fold IDs for validation (1-5)
            sr: Sample rate
            n_fft: FFT size
            hop: Hop length
            snr_range: (min_snr, max_snr) in dB
            cache_embeddings: Whether to cache CLAP text embeddings
        """
        self.esc50_path = Path(esc50_path)
        self.split = split
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.snr_range = snr_range
        
        # Load metadata
        meta_path = self.esc50_path / "meta" / "esc50.csv"
        self.df = pd.read_csv(meta_path)
        
        # Filter by fold
        fold_ids = train_fold_ids if split == "train" else val_fold_ids
        self.df = self.df[self.df['fold'].isin(fold_ids)].reset_index(drop=True)
        
        # Get unique categories
        self.categories = sorted(self.df['category'].unique())
        
        # Cache CLAP text embeddings for efficiency
        self.text_embeddings = {}
        if cache_embeddings:
            print(f"Caching text embeddings for {len(self.categories)} categories...")
            for cat in self.categories:
                self.text_embeddings[cat] = torch.from_numpy(
                    text_embed([cat])[0]
                ).float()
            print("Text embeddings cached!")
        
        print(f"{split.upper()} dataset: {len(self.df)} files from folds {fold_ids}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_clip(self, idx: int) -> Tuple[np.ndarray, str]:
        """Load audio clip and return waveform and category."""
        row = self.df.iloc[idx]
        audio_path = self.esc50_path / "audio" / row['filename']
        category = row['category']
        
        # Load audio
        y = load_audio(str(audio_path), sr=self.sr)
        
        # Ensure consistent length (5 seconds for ESC-50)
        target_length = 5 * self.sr
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        return y, category
    
    def _mix_at_snr(
        self,
        target: np.ndarray,
        interferer: np.ndarray,
        snr_db: float
    ) -> np.ndarray:
        """
        Mix two signals at specified SNR.
        
        SNR = 10 * log10(P_target / P_interferer)
        
        Args:
            target: Target signal
            interferer: Interferer signal
            snr_db: Desired SNR in dB
            
        Returns:
            Mixed signal
        """
        # Compute power
        p_target = np.mean(target ** 2)
        p_interferer = np.mean(interferer ** 2)
        
        # Compute scaling factor for interferer
        snr_linear = 10 ** (snr_db / 10.0)
        scale = np.sqrt(p_target / (p_interferer * snr_linear + 1e-8))
        
        # Mix
        mixture = target + scale * interferer
        
        # Normalize to prevent clipping
        max_val = np.abs(mixture).max()
        if max_val > 0.99:
            mixture = mixture * 0.99 / max_val
        
        return mixture
    
    def __getitem__(self, idx: int) -> dict:
        """
        Generate a training sample.
        
        Returns:
            Dictionary with:
            - mix_mag: Mixture magnitude spectrogram [1, F, T]
            - text_emb: CLAP text embedding [512]
            - target_mask: Ground truth IRM [1, F, T]
            - category: Target category name (for debugging)
        """
        # Load target clip
        target_wav, target_category = self._load_clip(idx)
        
        # Randomly select interferer (different from target)
        interferer_idx = random.randint(0, len(self.df) - 1)
        while interferer_idx == idx:
            interferer_idx = random.randint(0, len(self.df) - 1)
        interferer_wav, _ = self._load_clip(interferer_idx)
        
        # Random SNR
        snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Create mixture
        mixture_wav = self._mix_at_snr(target_wav, interferer_wav, snr_db)
        
        # Compute spectrograms
        _, target_mag, _ = stft(target_wav, n_fft=self.n_fft, hop=self.hop)
        _, interferer_mag, _ = stft(interferer_wav, n_fft=self.n_fft, hop=self.hop)
        _, mix_mag, _ = stft(mixture_wav, n_fft=self.n_fft, hop=self.hop)
        
        # Compute Ideal Ratio Mask
        target_mag_t = torch.from_numpy(target_mag).float()
        interferer_mag_t = torch.from_numpy(interferer_mag).float()
        target_mask = compute_irm(target_mag_t, interferer_mag_t)
        
        # Get text embedding
        if target_category in self.text_embeddings:
            text_emb = self.text_embeddings[target_category]
        else:
            text_emb = torch.from_numpy(text_embed([target_category])[0]).float()
        
        # Convert to tensors and add channel dimension
        mix_mag_t = torch.from_numpy(mix_mag).float().unsqueeze(0)  # [1, F, T]
        target_mask = target_mask.unsqueeze(0)  # [1, F, T]
        
        return {
            'mix_mag': mix_mag_t,
            'text_emb': text_emb,
            'target_mask': target_mask,
            'category': target_category
        }


def collate_fn(batch: List[dict]) -> dict:
    """Custom collate function to handle variable-length spectrograms."""
    # Find max dimensions
    max_freq = max(item['mix_mag'].shape[1] for item in batch)
    max_time = max(item['mix_mag'].shape[2] for item in batch)
    
    # Pad all spectrograms to max dimensions
    mix_mags = []
    target_masks = []
    text_embs = []
    categories = []
    
    for item in batch:
        mix_mag = item['mix_mag']
        target_mask = item['target_mask']
        
        # Pad frequency and time dimensions
        pad_freq = max_freq - mix_mag.shape[1]
        pad_time = max_time - mix_mag.shape[2]
        
        if pad_freq > 0 or pad_time > 0:
            mix_mag = torch.nn.functional.pad(
                mix_mag, (0, pad_time, 0, pad_freq), mode='constant', value=0
            )
            target_mask = torch.nn.functional.pad(
                target_mask, (0, pad_time, 0, pad_freq), mode='constant', value=0
            )
        
        mix_mags.append(mix_mag)
        target_masks.append(target_mask)
        text_embs.append(item['text_emb'])
        categories.append(item['category'])
    
    return {
        'mix_mag': torch.stack(mix_mags),
        'text_emb': torch.stack(text_embs),
        'target_mask': torch.stack(target_masks),
        'category': categories
    }


if __name__ == "__main__":
    # Test dataset
    esc50_path = Path(__file__).parent.parent / "data" / "ESC-50-master" / "ESC-50-master"
    
    dataset = ESC50MixtureDataset(
        esc50_path=str(esc50_path),
        split="train",
        cache_embeddings=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Categories: {len(dataset.categories)}")
    
    # Test sample
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  Mix magnitude shape: {sample['mix_mag'].shape}")
    print(f"  Text embedding shape: {sample['text_emb'].shape}")
    print(f"  Target mask shape: {sample['target_mask'].shape}")
    print(f"  Category: {sample['category']}")
    print(f"  Mask range: [{sample['target_mask'].min():.3f}, {sample['target_mask'].max():.3f}]")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  Mix magnitude shape: {batch['mix_mag'].shape}")
    print(f"  Text embedding shape: {batch['text_emb'].shape}")
    print(f"  Target mask shape: {batch['target_mask'].shape}")
    print(f"  Categories: {batch['category']}")
