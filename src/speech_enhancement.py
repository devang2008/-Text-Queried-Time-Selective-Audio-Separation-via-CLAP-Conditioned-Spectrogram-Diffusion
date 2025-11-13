"""
Speech enhancement module for improving separated speech clarity.
Includes: denoising, dereverberation, EQ, and loudness normalization.
"""
import numpy as np
import torch
from scipy import signal
import librosa


class SpeechEnhancer:
    """
    Simple speech enhancement pipeline for post-processing separated audio.
    """
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def spectral_denoise(self, audio, noise_profile_duration=0.5, reduction_factor=2.0):
        """
        Spectral subtraction denoising.
        
        Args:
            audio: numpy array [samples]
            noise_profile_duration: seconds to use for noise estimation
            reduction_factor: how aggressively to reduce noise
        
        Returns:
            Denoised audio
        """
        # Estimate noise from first portion
        noise_samples = int(noise_profile_duration * self.sr)
        noise_samples = min(noise_samples, len(audio) // 4)  # Use max 25% for noise
        
        if len(audio) < self.sr:  # Audio too short
            return audio
        
        noise = audio[:noise_samples]
        
        # Compute noise spectrum
        noise_stft = librosa.stft(noise, n_fft=2048, hop_length=512)
        noise_mag = np.abs(noise_stft).mean(axis=1, keepdims=True)
        
        # Compute audio spectrum
        audio_stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        audio_mag = np.abs(audio_stft)
        audio_phase = np.angle(audio_stft)
        
        # Spectral subtraction with over-subtraction
        enhanced_mag = np.maximum(audio_mag - reduction_factor * noise_mag, 0.1 * audio_mag)
        
        # Reconstruct
        enhanced_stft = enhanced_mag * np.exp(1j * audio_phase)
        enhanced = librosa.istft(enhanced_stft, hop_length=512, length=len(audio))
        
        return enhanced
    
    def dereverb(self, audio):
        """
        Simple dereverberation using high-pass filtering.
        
        Args:
            audio: numpy array [samples]
        
        Returns:
            Dereverberated audio
        """
        # High-pass filter to reduce low-frequency reverb
        sos = signal.butter(4, 80, 'hp', fs=self.sr, output='sos')
        derevered = signal.sosfilt(sos, audio)
        
        return derevered
    
    def speech_eq(self, audio):
        """
        Apply gentle EQ to enhance speech intelligibility.
        Boosts presence range (2-4 kHz).
        
        Args:
            audio: numpy array [samples]
        
        Returns:
            EQ'd audio
        """
        # Boost speech presence range (2-4 kHz)
        sos = signal.butter(2, [2000, 4000], 'bp', fs=self.sr, output='sos')
        boost = signal.sosfilt(sos, audio)
        
        # Mix with original (gentle boost)
        enhanced = audio + 0.25 * boost
        
        return enhanced
    
    def normalize_loudness(self, audio, target_lufs=-23):
        """
        Normalize loudness to target LUFS (using RMS approximation).
        
        Args:
            audio: numpy array [samples]
            target_lufs: target loudness in LUFS
        
        Returns:
            Normalized audio
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 1e-6:  # Silent audio
            return audio
        
        # Target RMS (approximation of LUFS)
        target_rms = 10 ** (target_lufs / 20)
        
        # Normalize
        normalized = audio * (target_rms / rms)
        
        # Prevent clipping
        max_val = np.abs(normalized).max()
        if max_val > 0.95:
            normalized = normalized * (0.95 / max_val)
        
        return normalized
    
    def enhance(self, audio, denoise=True, dereverb=True, eq=True, normalize=True):
        """
        Full enhancement pipeline.
        
        Args:
            audio: numpy array [samples] or torch.Tensor
            denoise: apply denoising
            dereverb: apply dereverberation
            eq: apply speech EQ
            normalize: apply loudness normalization
        
        Returns:
            Enhanced audio (same type as input)
        """
        # Convert to numpy if torch tensor
        input_is_torch = isinstance(audio, torch.Tensor)
        if input_is_torch:
            device = audio.device
            audio = audio.cpu().numpy()
        
        enhanced = audio.copy()
        
        # Step 1: Denoise
        if denoise:
            enhanced = self.spectral_denoise(enhanced)
        
        # Step 2: Dereverb
        if dereverb:
            enhanced = self.dereverb(enhanced)
        
        # Step 3: Speech EQ
        if eq:
            enhanced = self.speech_eq(enhanced)
        
        # Step 4: Normalize loudness
        if normalize:
            enhanced = self.normalize_loudness(enhanced)
        
        # Convert back to torch if needed
        if input_is_torch:
            enhanced = torch.from_numpy(enhanced).float().to(device)
        
        return enhanced


def enhance_speech_simple(audio, sr=16000):
    """
    Quick helper function for speech enhancement.
    
    Args:
        audio: numpy array or torch.Tensor [samples]
        sr: sample rate
    
    Returns:
        Enhanced audio (same type as input)
    """
    enhancer = SpeechEnhancer(sr=sr)
    return enhancer.enhance(audio)


# Example usage
if __name__ == "__main__":
    import soundfile as sf
    
    # Load audio
    audio, sr = sf.read("test_speech.wav")
    
    # Enhance
    enhancer = SpeechEnhancer(sr=sr)
    enhanced = enhancer.enhance(audio)
    
    # Save
    sf.write("test_speech_enhanced.wav", enhanced, sr)
    print("Enhanced audio saved!")
