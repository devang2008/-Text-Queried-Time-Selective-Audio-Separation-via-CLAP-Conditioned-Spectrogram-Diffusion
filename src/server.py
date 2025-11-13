"""
FastAPI server for audio separation application.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Literal
import base64
from pathlib import Path
import uvicorn
import os
import traceback
import shutil
import uuid

from config import list_audio_files, IMG_DIR, AUDIO_DIR, SAMPLE_RATE, STFT_N_FFT, STFT_HOP
from audio_utils import load_audio, stft, save_spectrogram_png
from nmf_sep import separate_with_text
from unet_sep import separate_with_unet, load_unet_model
from clap_embed import text_embed, audio_embed, cosine_sim, load_clap
from audio_analyzer import analyze_audio_with_gemini, get_quick_sound_suggestions

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = PROJECT_ROOT / "outputs" / "uploads"

app = FastAPI(title="Audio Separation API")

# Enable CORS for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directories on startup
@app.on_event("startup")
async def startup_event():
    Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
    Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    print("Server started. Output directories created.")
    # Preload CLAP model
    try:
        load_clap()
    except Exception as e:
        print(f"Warning: Could not preload CLAP model: {e}")
    # Preload UNet model
    try:
        load_unet_model()
    except Exception as e:
        print(f"Warning: Could not preload UNet model: {e}")
        print("  UNet separation will not be available. Train the model first!")


# Mount static files and outputs
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Global cache for audio files
_audio_files_cache = None

def get_audio_files():
    """Get audio files with caching."""
    global _audio_files_cache
    if _audio_files_cache is None:
        _audio_files_cache = list_audio_files()
    return _audio_files_cache

@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/api/files")
async def get_files():
    """
    Get list of available ESC-50 audio files.
    
    Returns:
        List of dicts with id, path, class_label
    """
    try:
        files = get_audio_files()
        # Don't expose full paths to client, but provide audio URL
        return [{"id": f["id"], "class_label": f["class_label"], "audio_url": f"/api/audio/{f['id']}"} for f in files]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@app.get("/api/audio/{file_id}")
async def get_audio(file_id: str):
    """
    Serve audio file by ID (dataset or uploaded).
    
    Args:
        file_id: File identifier
        
    Returns:
        Audio file
    """
    try:
        # Check if it's an uploaded file
        if file_id.startswith("upload_"):
            upload_files = list(UPLOADS_DIR.glob(f"{file_id}.*"))
            if not upload_files:
                raise HTTPException(status_code=404, detail="Uploaded file not found")
            return FileResponse(str(upload_files[0]), media_type="audio/wav")
        
        # Dataset file
        files = get_audio_files()
        file_info = next((f for f in files if f["id"] == file_id), None)
        
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        audio_path = Path(file_info["path"])
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found on disk")
        
        return FileResponse(str(audio_path), media_type="audio/wav")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving audio: {str(e)}")

@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload a custom audio file for separation.
    
    Args:
        file: Audio file (WAV, MP3, FLAC, etc.)
        
    Returns:
        File ID and metadata
    """
    try:
        # Validate file extension
        allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique ID and save file
        file_id = f"upload_{uuid.uuid4().hex[:8]}"
        upload_path = UPLOADS_DIR / f"{file_id}{file_ext}"
        
        # Save uploaded file
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load audio to validate and get duration
        try:
            y = load_audio(str(upload_path), sr=SAMPLE_RATE)
            duration = len(y) / SAMPLE_RATE
        except Exception as e:
            # Clean up invalid file
            upload_path.unlink()
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")
        
        return {
            "id": file_id,
            "filename": file.filename,
            "path": str(upload_path),
            "duration": round(duration, 2),
            "sample_rate": SAMPLE_RATE,
            "uploaded": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/api/analyze")
async def analyze_audio_content(file_id: str):
    """
    Analyze audio content using Gemini API to detect sound classes.
    
    Args:
        file_id: File identifier (uploaded file)
        
    Returns:
        JSON with detected sounds, characteristics, and separation suggestions
    """
    try:
        # Check if it's an uploaded file
        if file_id.startswith("upload_"):
            upload_files = list(UPLOADS_DIR.glob(f"{file_id}.*"))
            if not upload_files:
                raise HTTPException(status_code=404, detail="Uploaded file not found")
            audio_path = str(upload_files[0])
        else:
            raise HTTPException(
                status_code=400, 
                detail="Analysis only available for uploaded files. Dataset files already have class labels."
            )
        
        # Analyze with Gemini
        result = analyze_audio_with_gemini(audio_path)
        
        if not result.get("success"):
            error_msg = result.get("message", result.get("error", "Analysis failed"))
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Return only specific sounds
        return {
            "file_id": file_id,
            "specific_sounds": result.get("specific_sounds", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/api/spectrogram")
async def get_spectrogram(file_id: str):
    """
    Generate and return base64-encoded spectrogram PNG for a file.
    
    Args:
        file_id: File identifier (from dataset or uploaded)
        
    Returns:
        JSON with base64 encoded PNG
    """
    try:
        # Check if it's an uploaded file
        if file_id.startswith("upload_"):
            # Find uploaded file
            upload_files = list(UPLOADS_DIR.glob(f"{file_id}.*"))
            if not upload_files:
                raise HTTPException(status_code=404, detail="Uploaded file not found")
            audio_path = str(upload_files[0])
            label = "Uploaded Audio"
        else:
            # Dataset file
            files = get_audio_files()
            file_info = next((f for f in files if f["id"] == file_id), None)
            
            if not file_info:
                raise HTTPException(status_code=404, detail="File not found")
            
            audio_path = file_info["path"]
            label = file_info['class_label']
        
        # Check if spectrogram already exists
        spec_path = Path(IMG_DIR) / f"{file_id}_spectrogram.png"
        
        if not spec_path.exists():
            # Generate spectrogram
            y = load_audio(audio_path, sr=SAMPLE_RATE)
            _, mag, _ = stft(y, n_fft=STFT_N_FFT, hop=STFT_HOP)
            save_spectrogram_png(mag, str(spec_path), f"Spectrogram: {label}")
        
        # Return relative path for client to fetch
        return {"url": f"/outputs/img/{spec_path.name}"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating spectrogram: {str(e)}")

class SeparationRequest(BaseModel):
    file_id: str
    prompt: str
    mode: Literal["keep", "remove"] = "keep"
    t0: float = 0.0
    t1: Optional[float] = None
    k_components: int = 10

@app.post("/api/separate")
async def separate_audio(request: SeparationRequest):
    """
    Perform audio separation using NMF+CLAP (legacy method).
    
    Args:
        request: Separation parameters
        
    Returns:
        JSON with output file URLs and confidence score
    """
    try:
        # Check if it's an uploaded file
        if request.file_id.startswith("upload_"):
            # Find uploaded file
            upload_files = list(UPLOADS_DIR.glob(f"{request.file_id}.*"))
            if not upload_files:
                raise HTTPException(status_code=404, detail="Uploaded file not found")
            audio_path = str(upload_files[0])
        else:
            # Dataset file
            files = get_audio_files()
            file_info = next((f for f in files if f["id"] == request.file_id), None)
            
            if not file_info:
                raise HTTPException(status_code=404, detail="File not found")
            
            audio_path = file_info["path"]
        
        # Validate parameters
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if request.t1 is not None and request.t1 <= request.t0:
            raise HTTPException(status_code=400, detail="End time must be greater than start time")
        
        # Run separation
        result = separate_with_text(
            audio_path=audio_path,
            prompt=request.prompt,
            mode=request.mode,
            k_components=request.k_components,
            t0=request.t0,
            t1=request.t1
        )
        
        # Convert file paths to URLs
        response_data = {
            "out_wav": f"/outputs/audio/{Path(result['out_wav']).name}",
            "residual_wav": f"/outputs/audio/{Path(result['residual_wav']).name}",
            "mask_png": f"/outputs/img/{Path(result['mask_png']).name}",
            "mix_spec_png": f"/outputs/img/{Path(result['mix_spec_png']).name}",
            "out_spec_png": f"/outputs/img/{Path(result['out_spec_png']).name}",
            "confidence": result["confidence"]
        }
        
        print(f"Separation response URLs: {response_data}")  # Debug logging
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Separation error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Separation error: {str(e)}")


@app.post("/api/separate_unet")
async def separate_audio_unet(request: SeparationRequest):
    """
    Perform audio separation using trained UNet model (NEW - trained model).
    
    Args:
        request: Separation parameters
        
    Returns:
        JSON with output file URLs and confidence score
    """
    try:
        # Check if it's an uploaded file
        if request.file_id.startswith("upload_"):
            # Find uploaded file
            upload_files = list(UPLOADS_DIR.glob(f"{request.file_id}.*"))
            if not upload_files:
                raise HTTPException(status_code=404, detail="Uploaded file not found")
            audio_path = str(upload_files[0])
        else:
            # Dataset file
            files = get_audio_files()
            file_info = next((f for f in files if f["id"] == request.file_id), None)
            
            if not file_info:
                raise HTTPException(status_code=404, detail="File not found")
            
            audio_path = file_info["path"]
        
        # Validate parameters
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if request.t1 is not None and request.t1 <= request.t0:
            raise HTTPException(status_code=400, detail="End time must be greater than start time")
        
        # Run UNet separation
        result = separate_with_unet(
            audio_path=audio_path,
            prompt=request.prompt,
            mode=request.mode,
            t0=request.t0,
            t1=request.t1,
            fade_ms=70.0
        )
        
        # Convert file paths to URLs
        response_data = {
            "out_wav": f"/outputs/audio/{Path(result['out_wav']).name}",
            "residual_wav": f"/outputs/audio/{Path(result['residual_wav']).name}",
            "mask_png": f"/outputs/img/{Path(result['mask_png']).name}",
            "mix_spec_png": f"/outputs/img/{Path(result['mix_spec_png']).name}",
            "out_spec_png": f"/outputs/img/{Path(result['out_spec_png']).name}",
            "confidence": result["confidence"],
            "method": "UNet (Trained Model)"
        }
        
        print(f"UNet separation response: {response_data}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"UNet separation error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"UNet separation error: {str(e)}")


class DetectClassesRequest(BaseModel):
    file_id: str
    k_components: int = 10

# Extended sound class prompts for detection
# Includes ESC-50 classes + common real-world sounds
ESC50_CLASS_PROMPTS = [
    # ESC-50 Animal sounds
    "dog barking",
    "rooster crowing",
    "pig",
    "cow",
    "frog",
    "cat",
    "hen",
    "insects",
    "sheep",
    "crow",
    
    # ESC-50 Natural sounds
    "rain falling",
    "sea waves",
    "fire crackling",
    "crickets",
    "thunderstorm",
    "wind blowing",
    "water drops",
    "pouring water",
    
    # ESC-50 Human sounds
    "baby crying",
    "sneezing",
    "clapping",
    "breathing",
    "coughing",
    "footsteps",
    "laughing",
    "snoring",
    
    # ESC-50 Interior sounds
    "door knock",
    "mouse click",
    "keyboard typing",
    "door creaking",
    "washing machine",
    "vacuum cleaner",
    "clock alarm",
    "clock ticking",
    "glass breaking",
    
    # ESC-50 Exterior sounds
    "helicopter",
    "chainsaw",
    "siren",
    "car horn",
    "engine",
    "train",
    "church bells",
    "airplane",
    "fireworks",
    
    # Common additional sounds (not in ESC-50)
    "music",
    "piano",
    "guitar",
    "drums",
    "singing",
    "speech",
    "male voice",
    "female voice",
    "applause",
    "crowd noise",
    "birds chirping",
    "ocean",
    "waterfall",
    "thunder",
    "gunshot",
    "explosion",
    "telephone ring",
    "notification sound",
    "alarm beep",
    "buzzer"
]

@app.post("/api/classes")
async def detect_classes(request: DetectClassesRequest):
    """
    Detect top-K sound classes in an audio file.
    
    Args:
        request: File ID and parameters
        
    Returns:
        List of detected classes with scores
    """
    try:
        files = get_audio_files()
        file_info = next((f for f in files if f["id"] == request.file_id), None)
        
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Load audio and get embedding
        y = load_audio(file_info["path"], sr=SAMPLE_RATE)
        audio_emb = audio_embed([y], [SAMPLE_RATE])  # [1, D]
        
        # Get text embeddings for class prompts
        text_embs = text_embed(ESC50_CLASS_PROMPTS)  # [N, D]
        
        # Compute similarities
        similarities = cosine_sim(audio_emb, text_embs)[0]  # [N]
        
        # Sort by similarity
        sorted_indices = similarities.argsort()[::-1]
        
        # Confidence threshold - only show if reasonably confident
        CONFIDENCE_THRESHOLD = 0.15  # Minimum similarity score to report
        
        # Return top 5 with confidence filtering
        results = []
        for idx in sorted_indices[:10]:  # Check top 10
            score = float(similarities[idx])
            if score >= CONFIDENCE_THRESHOLD:
                results.append({
                    "class": ESC50_CLASS_PROMPTS[idx],
                    "score": score
                })
            if len(results) >= 5:  # Return max 5 results
                break
        
        # If no confident matches, return message
        if not results:
            results.append({
                "class": "Unknown sound (no confident matches)",
                "score": float(similarities[sorted_indices[0]])
            })
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Detection error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
