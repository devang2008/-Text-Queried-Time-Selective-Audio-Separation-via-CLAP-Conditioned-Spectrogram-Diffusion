# Time-Selective, Text-Guided Audio Separation System

A student-friendly machine learning system for audio editing that allows users to isolate ("keep") or suppress ("remove") specific sounds from audio mixtures, guided by natural language text prompts. The key novelty is **time-selective editing** - edits are applied only within user-specified time windows, leaving the rest of the audio untouched.

## 🎯 Project Overview

This project implements a complete pipeline for text-guided audio separation with the following features:

- **Text-Guided Separation**: Use natural language (e.g., "bird chirps", "female speech") to describe the target sound
- **Time-Selective Editing**: Apply separation only within specified time windows `[start_sec, end_sec]`
- **Keep or Remove Modes**: Isolate the target sound or suppress it
- **UNet-based Model**: Trainable deep learning model with CLAP text conditioning
- **Web Interface**: Easy-to-use web application for testing

## 🏗️ System Architecture

The system consists of three main stages:

### 1. Text-Audio Preprocessing
- **Text Embedding (CLAP)**: Converts text prompts to 512-dim embeddings using pretrained CLAP model
- **Audio Representation**: Converts audio to magnitude + phase spectrograms via STFT

### 2. Separation Model (Trainable UNet)
- **Architecture**: UNet with FiLM (Feature-wise Linear Modulation) layers for text conditioning
- **Input**: Mixture magnitude spectrogram + CLAP text embedding
- **Output**: Soft mask [0, 1] indicating time-frequency bins of target sound
- **Training**: Supervised learning on synthetic mixtures from ESC-50 dataset

### 3. Time-Gated Synthesis
- **Time Gate Creation**: 1D gate vector with smooth fade-in/fade-out at window boundaries
- **Mask Application**: Multiply predicted mask by time gate for selective editing
- **Audio Reconstruction**: ISTFT with original phase, blend with input audio outside windows

## 📁 Project Structure

```
CP_new/
├── src/
│   ├── model.py              # UNet architecture with text conditioning
│   ├── dataset.py            # ESC-50 mixture dataset with IRM ground truth
│   ├── train.py              # Training script
│   ├── inference.py          # Time-selective separation pipeline
│   ├── audio_utils.py        # STFT/ISTFT and time gating utilities
│   ├── clap_embed.py         # CLAP text/audio embeddings
│   ├── nmf_sep.py            # Legacy NMF-based separation (baseline)
│   ├── server.py             # FastAPI web server
│   └── config.py             # Configuration constants
├── static/
│   ├── index.html            # Web interface
│   ├── app.js               # Frontend JavaScript
│   └── styles.css           # CSS styling
├── data/
│   └── ESC-50-master/       # ESC-50 dataset
├── checkpoints/             # Saved model weights (created during training)
├── outputs/                 # Output audio and spectrograms
└── requirements.txt         # Python dependencies
```

## 🚀 Getting Started

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.0.0` - Deep learning framework
- `librosa` - Audio processing
- `msclap` - CLAP model for text-audio embeddings
- `fastapi` - Web API framework
- `pandas` - Data handling

### 2. Download ESC-50 Dataset

The ESC-50 dataset should already be in `data/ESC-50-master/ESC-50-master/`. If not:

```powershell
# Download from https://github.com/karolpiczak/ESC-50
# Extract to data/ESC-50-master/
```

### 3. Train the Model

Train the text-conditioned UNet model:

```powershell
cd src
python train.py --esc50-path ../data/ESC-50-master/ESC-50-master --epochs 50 --batch-size 8
```

Training options:
- `--baseline`: Train text-agnostic baseline (for comparison)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-3)
- `--train-folds`: ESC-50 folds for training (default: [1,2,3,4])
- `--val-folds`: ESC-50 folds for validation (default: [5])
- `--output-dir`: Where to save checkpoints (default: ../checkpoints)

### 4. Run Inference

Use the trained model for time-selective separation:

```powershell
python inference.py \
    --audio-in path/to/mixture.wav \
    --audio-out path/to/output.wav \
    --prompt-text "dog bark" \
    --mode keep \
    --time-windows 0 4 \
    --model-path ../checkpoints/text_conditioned_unet_best.pth
```

#### Inference Arguments

**Required:**
- `--audio-in`: Input audio file (Audio_In)
- `--audio-out`: Output audio file (Audio_Out)
- `--prompt-text`: Text description of target sound (e.g., "dog bark", "female speech")
- `--time-windows`: Time windows as pairs `t0_1 t1_1 t0_2 t1_2 ...` (in seconds)
- `--mode`: `keep` or `remove`
- `--model-path`: Path to trained model checkpoint

**Optional:**
- `--baseline`: Use baseline model (text-agnostic)
- `--fade-ms`: Fade duration in milliseconds (default: 70)
- `--cpu`: Force CPU inference

#### Examples

**Keep dog bark from 0-4 seconds:**
```powershell
python inference.py --audio-in mix.wav --audio-out out.wav \
    --prompt-text "dog bark" --mode keep --time-windows 0 4 \
    --model-path ../checkpoints/text_conditioned_unet_best.pth
```

**Remove speech from multiple time windows:**
```powershell
python inference.py --audio-in mix.wav --audio-out out.wav \
    --prompt-text "speech" --mode remove --time-windows 2 6 8 10 \
    --model-path ../checkpoints/text_conditioned_unet_best.pth
```

### 5. Web Interface

Start the web server:

```powershell
python server.py
```

Open browser to `http://localhost:8000`

Features:
- Browse ESC-50 dataset files
- Upload custom audio files
- Detect sound classes with CLAP
- AI analysis with Gemini API
- Separate audio with text prompts

## 📊 Evaluation Metrics

The system is evaluated using window-aware metrics:

### Inside the Window:
1. **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio)
2. **SDR** (Signal-to-Distortion Ratio)
3. **Text-Audio Similarity**: Cosine similarity between separated audio and text prompt using CLAP

### Outside the Window:
1. **No-Change Error**: L2 difference between input and output (should be ~0)
2. **SI-SDR**: Should be near infinite (perfect preservation)

### Baselines:
1. **Text-Agnostic Baseline**: Same UNet without text conditioning
2. **No-Gate Baseline**: Apply mask to entire audio (no time-selectivity)

## 🔧 Configuration

Edit `src/config.py` to modify:
- `SAMPLE_RATE`: Audio sample rate (default: 16000 Hz)
- `STFT_N_FFT`: FFT window size (default: 1024)
- `STFT_HOP`: Hop length (default: 256)
- ESC-50 dataset path

## 📝 Model Architecture

### TextConditionedUNet
- **Encoder**: 4 downsampling blocks (64 → 128 → 256 → 512 channels)
- **Bottleneck**: 512 channels with FiLM conditioning
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Soft mask with sigmoid activation [0, 1]
- **Conditioning**: FiLM layers inject CLAP text embeddings into decoder blocks

### Training Details
- **Loss Function**: L1 loss between predicted mask and Ideal Ratio Mask (IRM)
- **Optimizer**: Adam with learning rate 1e-3
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Data Augmentation**: Random SNR mixing (-5dB to +5dB)

## 🎵 Dataset: ESC-50

- **Size**: 2000 audio clips (5 seconds each)
- **Categories**: 50 environmental sound classes
- **Organization**: 5 cross-validation folds
- **Training Strategy**: On-the-fly synthetic mixture generation
  - Randomly select target and interferer clips
  - Mix at random SNR
  - Compute Ideal Ratio Mask (IRM) as ground truth

## 📈 Training Progress

Training logs are saved to `checkpoints/{model_name}_history.json`:
```json
{
  "train_loss": [...],
  "val_loss": [...],
  "lr": [...]
}
```

Checkpoints saved:
- `{model_name}_best.pth`: Best model based on validation loss
- `{model_name}_epoch{N}.pth`: Checkpoint every N epochs
- `{model_name}_final.pth`: Final model after all epochs

## 🧪 Testing the System

### Quick Test
```powershell
# 1. Train for a few epochs (quick test)
python train.py --epochs 5 --batch-size 4

# 2. Create test mixture
# (Mix two ESC-50 files manually or use web interface)

# 3. Run inference
python inference.py \
    --audio-in test_mix.wav \
    --audio-out test_out.wav \
    --prompt-text "dog" \
    --mode keep \
    --time-windows 0 5 \
    --model-path ../checkpoints/text_conditioned_unet_best.pth
```

## 🐛 Troubleshooting

### CLAP Model Issues
```powershell
# Install msclap
pip install msclap

# Or use laion-clap alternative
pip install laion-clap
```

### CUDA/GPU Issues
```powershell
# Force CPU training
python train.py --cpu

# Force CPU inference
python inference.py --cpu ...
```

### Memory Issues
```powershell
# Reduce batch size
python train.py --batch-size 2

# Use fewer workers
python train.py --num-workers 0
```

## 📚 Key References

1. **CLAP**: Contrastive Language-Audio Pretraining
   - Paper: https://arxiv.org/abs/2211.06687
   - Model: `msclap` or `laion-clap`

2. **UNet**: Convolutional Networks for Biomedical Image Segmentation
   - Adapted for audio spectrograms with text conditioning

3. **FiLM**: Feature-wise Linear Modulation
   - Used for injecting text conditioning into UNet

4. **ESC-50**: Dataset for Environmental Sound Classification
   - Paper: https://github.com/karolpiczak/ESC-50

## 🎓 Learning Objectives

This project demonstrates:
1. **Deep Learning for Audio**: Spectrogram-based processing with UNet
2. **Multimodal Learning**: Combining text and audio with CLAP
3. **Conditional Generation**: Using FiLM layers for controllable separation
4. **Audio Signal Processing**: STFT/ISTFT, masking, time gating
5. **Full-Stack ML**: Training, inference, evaluation, and web deployment

## 🔮 Future Improvements

1. **Diffusion Model**: Implement denoising diffusion for better quality
2. **Speech Enhancement**: Add post-processing for speech separation
3. **Multi-source Separation**: Separate multiple sounds simultaneously
4. **Real-time Processing**: Optimize for streaming audio
5. **Mobile Deployment**: Export to ONNX/TorchScript

## 📄 License

This project is for educational purposes.

## 👥 Contributors

TY-SEM-I ML Course Project

---

**Happy Separating! 🎵**
