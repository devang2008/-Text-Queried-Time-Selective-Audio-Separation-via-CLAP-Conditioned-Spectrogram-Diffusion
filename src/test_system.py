"""
Quick test script to verify all components work before training.
"""
print("="*60)
print("TESTING AUDIO SEPARATION SYSTEM COMPONENTS")
print("="*60)

print("\n1. Testing imports...")
try:
    import torch
    import numpy as np
    from model import TextConditionedUNet, TextAgnosticUNet
    from dataset import ESC50MixtureDataset
    from audio_utils import load_audio, stft, istft, time_gate
    from clap_embed import load_clap, text_embed
    print("   ✓ All imports successful!")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

print("\n2. Testing CLAP model...")
try:
    model = load_clap()
    test_emb = text_embed(["dog bark"])
    print(f"   ✓ CLAP loaded! Embedding shape: {test_emb.shape}")
except Exception as e:
    print(f"   ✗ CLAP failed: {e}")
    exit(1)

print("\n3. Testing UNet model...")
try:
    unet = TextConditionedUNet(clap_dim=1024)
    params = sum(p.numel() for p in unet.parameters())
    print(f"   ✓ UNet created! Parameters: {params:,}")
    
    # Test forward pass
    dummy_mag = torch.randn(1, 1, 513, 313)
    dummy_text = torch.randn(1, 1024)
    mask = unet(dummy_mag, dummy_text)
    print(f"   ✓ Forward pass works! Mask shape: {mask.shape}")
    print(f"   ✓ Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
except Exception as e:
    print(f"   ✗ UNet failed: {e}")
    exit(1)

print("\n4. Testing dataset...")
try:
    dataset = ESC50MixtureDataset(
        esc50_path='../data/ESC-50-master/ESC-50-master',
        split='train',
        cache_embeddings=False  # Fast test without caching
    )
    print(f"   ✓ Dataset loaded! Size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"   ✓ Sample loaded!")
    print(f"     - Mix mag: {sample['mix_mag'].shape}")
    print(f"     - Text emb: {sample['text_emb'].shape}")
    print(f"     - Target mask: {sample['target_mask'].shape}")
    print(f"     - Category: {sample['category']}")
except Exception as e:
    print(f"   ✗ Dataset failed: {e}")
    exit(1)

print("\n5. Testing audio utils...")
try:
    # Load a test audio file
    test_audio_path = '../data/ESC-50-master/ESC-50-master/audio/1-100032-A-0.wav'
    audio = load_audio(test_audio_path, sr=16000)
    print(f"   ✓ Audio loaded! Shape: {audio.shape}")
    
    # Test STFT/ISTFT
    _, mag, phase = stft(audio)
    reconstructed = istft(mag, phase)
    print(f"   ✓ STFT/ISTFT works! Spectrogram: {mag.shape}")
    
    # Test time gate
    gate = time_gate(n_frames=313, sr=16000, hop=256, t0=1.0, t1=3.0, fade_ms=70)
    print(f"   ✓ Time gate works! Shape: {gate.shape}")
except Exception as e:
    print(f"   ✗ Audio utils failed: {e}")
    exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nYour system is ready to train!")
print("\nNext steps:")
print("1. Run: ./quick_train.bat")
print("   OR:   cd src && python train.py --epochs 5 --batch-size 4")
print("\n2. After training, run inference:")
print("   cd src && python inference.py \\")
print("       --audio-in test.wav \\")
print("       --audio-out output.wav \\")
print("       --prompt-text 'dog bark' \\")
print("       --mode keep \\")
print("       --time-windows 0 5 \\")
print("       --model-path ../checkpoints/text_conditioned_unet_best.pth")
print("\n3. Or use the web interface:")
print("   python server.py")
print("   Then open: http://localhost:8000")
print("="*60)
