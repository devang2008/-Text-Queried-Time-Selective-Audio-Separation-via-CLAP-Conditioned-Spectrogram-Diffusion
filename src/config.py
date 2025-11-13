"""
Configuration file for ESC-50 audio separation project.
Contains paths, audio parameters, and helper functions.
"""
import csv
from pathlib import Path
from typing import List, Dict

# Data paths (Windows path-safe)
DATA_ROOT = Path(r"C:\Users\devan\OneDrive\Desktop\TY-SEM-I\ML\CP_new\data\ESC-50-master\ESC-50-master")
AUDIO_DIR_SRC = DATA_ROOT / "audio"
META_FILE = DATA_ROOT / "meta" / "esc50.csv"

# Audio processing parameters
SAMPLE_RATE = 16000
STFT_N_FFT = 1024
STFT_HOP = 256

# Output directories (use absolute paths relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from src/ to project root
IMG_DIR = str(PROJECT_ROOT / "outputs" / "img")
AUDIO_DIR = str(PROJECT_ROOT / "outputs" / "audio")

def list_audio_files() -> List[Dict[str, str]]:
    """
    Returns a list of audio files from ESC-50 dataset.
    
    Returns:
        List of dicts with keys: id, path, class_label
    """
    audio_files = []
    
    # Read metadata CSV
    if not META_FILE.exists():
        print(f"Warning: Meta file not found at {META_FILE}")
        return []
    
    with open(META_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            audio_path = AUDIO_DIR_SRC / filename
            
            if audio_path.exists():
                audio_files.append({
                    'id': filename.replace('.wav', ''),
                    'path': str(audio_path),
                    'class_label': row['category']
                })
    
    return audio_files

# Create output directories on import
Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)
