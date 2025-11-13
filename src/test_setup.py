"""
Test script to verify the installation and configuration.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import fastapi
        print("✓ FastAPI")
    except ImportError as e:
        print(f"✗ FastAPI: {e}")
        return False
    
    try:
        import librosa
        print("✓ Librosa")
    except ImportError as e:
        print(f"✗ Librosa: {e}")
        return False
    
    try:
        import soundfile
        print("✓ SoundFile")
    except ImportError as e:
        print(f"✗ SoundFile: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn")
    except ImportError as e:
        print(f"✗ Scikit-learn: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch")
        cuda_available = torch.cuda.is_available()
        print(f"  - CUDA available: {cuda_available}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        # Try msclap first
        import msclap
        print("✓ CLAP (msclap)")
    except ImportError:
        try:
            # Try laion-clap
            import laion_clap
            print("✓ CLAP (laion-clap)")
        except ImportError as e:
            print(f"✗ CLAP: {e}")
            print("  Please install: pip install msclap")
            return False
    
    return True

def test_data_path():
    """Test that ESC-50 data path exists."""
    print("\nTesting data path...")
    
    from config import DATA_ROOT, AUDIO_DIR_SRC, META_FILE
    
    if not DATA_ROOT.exists():
        print(f"✗ Data root not found: {DATA_ROOT}")
        return False
    print(f"✓ Data root: {DATA_ROOT}")
    
    if not AUDIO_DIR_SRC.exists():
        print(f"✗ Audio directory not found: {AUDIO_DIR_SRC}")
        return False
    print(f"✓ Audio directory: {AUDIO_DIR_SRC}")
    
    if not META_FILE.exists():
        print(f"✗ Metadata file not found: {META_FILE}")
        return False
    print(f"✓ Metadata file: {META_FILE}")
    
    # Count audio files
    audio_files = list(AUDIO_DIR_SRC.glob("*.wav"))
    print(f"  - Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("✗ No audio files found!")
        return False
    
    return True

def test_output_dirs():
    """Test that output directories exist."""
    print("\nTesting output directories...")
    
    from config import IMG_DIR, AUDIO_DIR
    
    img_path = Path(IMG_DIR)
    audio_path = Path(AUDIO_DIR)
    
    if img_path.exists():
        print(f"✓ Image output directory: {img_path}")
    else:
        print(f"✗ Image output directory not found: {img_path}")
        return False
    
    if audio_path.exists():
        print(f"✓ Audio output directory: {audio_path}")
    else:
        print(f"✗ Audio output directory not found: {audio_path}")
        return False
    
    return True

def test_static_files():
    """Test that static files exist."""
    print("\nTesting static files...")
    
    # Go up one level from src/ to find static/
    static_path = Path(__file__).parent.parent / "static"
    
    files = ["index.html", "styles.css", "app.js"]
    all_exist = True
    
    for filename in files:
        file_path = static_path / filename
        if file_path.exists():
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} not found at {static_path}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("=" * 50)
    print("Audio Separation Setup Test")
    print("=" * 50)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Data Path", test_data_path()))
    results.append(("Output Directories", test_output_dirs()))
    results.append(("Static Files", test_static_files()))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n✓ All tests passed! You can now run the server.")
        print("\nTo start the server:")
        print("  1. Run: python server.py")
        print("  2. Open: http://localhost:8000")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
