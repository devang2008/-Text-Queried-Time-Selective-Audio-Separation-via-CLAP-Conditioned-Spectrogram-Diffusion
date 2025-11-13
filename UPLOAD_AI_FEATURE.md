# 🎵 Audio Upload & AI Analysis Feature - Implementation Summary

## ✨ What's New

Your audio separation app now supports:

1. **📤 Upload Custom Audio Files** - Upload your own mixed audio (not limited to ESC-50 dataset)
2. **🤖 AI-Powered Sound Detection** - Gemini AI automatically analyzes uploaded audio to detect sounds
3. **💡 Smart Separation Suggestions** - AI recommends text prompts optimized for separating detected sounds
4. **🎯 One-Click Prompt Fill** - Click suggested prompts to instantly use them for separation

## 📦 Files Added/Modified

### New Files Created:
- ✅ `src/gemini_audio_analyzer.py` - Gemini API integration for audio analysis
- ✅ `GEMINI_SETUP.md` - Complete setup guide for Gemini API

### Files Modified:
- ✅ `src/server.py` - Added upload endpoint, analyze endpoint, updated audio serving
- ✅ `static/index.html` - Added upload button, AI analysis section, results display
- ✅ `static/styles.css` - Styled upload section, analysis results, prompt suggestions
- ✅ `static/app.js` - Upload logic, AI analysis function, prompt auto-fill
- ✅ `requirements.txt` - Added `google-generativeai` dependency

## 🚀 How It Works

### User Flow:

```
1. User uploads mixed audio (dog + car + music)
   ↓
2. App saves file to outputs/uploads/
   ↓
3. User clicks "🤖 Analyze with Gemini AI"
   ↓
4. Gemini API analyzes audio:
   - Detects: dog barking, car engine, piano music
   - Suggests prompts: "dog barking", "car engine sound", etc.
   ↓
5. User clicks suggested prompt (e.g., "dog barking")
   ↓
6. Prompt auto-fills in separation box
   ↓
7. User runs separation to extract/remove that sound
```

### Technical Flow:

```
Frontend (app.js)
  │
  ├─ handleFileUpload()
  │   └─ POST /api/upload → saves to outputs/uploads/
  │
  ├─ analyzeAudioWithAI()
  │   └─ POST /api/analyze?file_id=upload_xxx
  │       └─ gemini_audio_analyzer.py
  │           └─ Gemini 1.5 Flash API
  │               └─ Returns: classes, sounds, prompts
  │
  └─ usePrompt(text)
      └─ Auto-fills promptInput field
```

## 🎨 UI Changes

### Left Panel (File Selection):
```
┌─────────────────────────────────┐
│  📁 Select Audio File           │
├─────────────────────────────────┤
│  [📤 Upload Your Audio]         │  ← NEW
│  ✅ Uploaded: mixed_audio.wav   │  ← NEW
│                                 │
│  ────────── OR ──────────       │
│                                 │
│  [▼ Select from Dataset...]     │
│                                 │
│  🎵 Input Audio                 │
│  [▶ Audio Player]               │
│  Class: Uploaded                │
│                                 │
│  [🤖 Analyze with Gemini AI]    │  ← NEW
│                                 │
│  ┌──── AI Audio Analysis ─────┐│  ← NEW
│  │ 📊 Main Classes:            ││
│  │ • speech • music            ││
│  │                             ││
│  │ 🔊 Detected Sounds:         ││
│  │ • dog barking • car engine  ││
│  │                             ││
│  │ 💡 Suggested Prompts:       ││
│  │ ▸ dog barking  ← clickable  ││
│  │ ▸ car engine                ││
│  └─────────────────────────────┘│
│                                 │
│  📊 Input Spectrogram           │
│  [Spectrogram Image]            │
└─────────────────────────────────┘
```

## 🔑 Gemini API Setup (IMPORTANT!)

### Quick Setup:

1. **Get API Key:**
   - Visit: https://aistudio.google.com/app/apikey
   - Click "Get API Key"
   - Copy your key

2. **Add to Your Project:**

   **Option A - Environment Variable (Recommended):**
   ```powershell
   $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
   python src/server.py
   ```

   **Option B - Direct in Code:**
   ```python
   # Edit src/gemini_audio_analyzer.py line 10:
   GEMINI_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
   ```

3. **Install SDK:**
   ```bash
   pip install google-generativeai
   ```

4. **Test:**
   ```bash
   python src/gemini_audio_analyzer.py path/to/audio.wav
   ```

**See GEMINI_SETUP.md for complete instructions!**

## 📊 API Endpoints Added

### POST /api/upload
Upload audio file for separation.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (audio file)

**Response:**
```json
{
  "id": "upload_a1b2c3d4",
  "filename": "mixed_audio.wav",
  "path": "C:/path/to/uploads/upload_a1b2c3d4.wav",
  "duration": 15.3,
  "sample_rate": 16000,
  "uploaded": true
}
```

**Supported Formats:** WAV, MP3, FLAC, OGG, M4A

---

### POST /api/analyze
Analyze uploaded audio with Gemini AI.

**Request:**
- Method: `POST`
- Query Param: `file_id=upload_xxx`

**Response:**
```json
{
  "file_id": "upload_a1b2c3d4",
  "main_classes": ["speech", "music", "nature sounds"],
  "specific_sounds": ["male voice", "piano", "bird chirping"],
  "characteristics": "Clear speech with soft piano background...",
  "separation_prompts": [
    "male voice speaking",
    "piano music",
    "bird chirping",
    "background music",
    "speech without music"
  ],
  "raw_analysis": "Full AI response text..."
}
```

**Error Responses:**
- `400` - Not an uploaded file (dataset files don't need analysis)
- `404` - File not found
- `500` - Gemini API error (check API key)

---

### Updated: GET /api/audio/{file_id}
Now supports uploaded files.

**Behavior:**
- If `file_id` starts with `upload_` → serves from `outputs/uploads/`
- Otherwise → serves from ESC-50 dataset

---

### Updated: GET /api/spectrogram?file_id={id}
Generates spectrograms for uploaded files.

**Behavior:**
- Detects uploaded vs dataset files
- Generates spectrogram if not cached
- Returns URL: `/outputs/img/upload_xxx_spectrogram.png`

---

### Updated: POST /api/separate
Separates audio from uploaded files.

**Changes:**
- Accepts `file_id` starting with `upload_`
- Finds file in `outputs/uploads/`
- Runs NMF+CLAP separation
- Returns output/residual audio + spectrograms

## 🧪 Testing

### Test 1: Upload Feature
```bash
# Start server
cd src
python server.py

# Open browser: http://localhost:8000
# 1. Click "Upload Your Audio"
# 2. Select an audio file
# 3. Verify: "✅ Uploaded: filename.wav" appears
# 4. Verify: Audio plays
# 5. Verify: Spectrogram displays
```

### Test 2: AI Analysis (Requires API Key)
```bash
# Set API key
$env:GEMINI_API_KEY="your_key_here"

# Start server
python src/server.py

# In browser:
# 1. Upload audio file
# 2. Click "🤖 Analyze with Gemini AI"
# 3. Wait 5-15 seconds
# 4. Verify: AI analysis appears with:
#    - Main classes
#    - Specific sounds
#    - Characteristics
#    - Suggested prompts
```

### Test 3: Separation with Uploaded File
```bash
# In browser:
# 1. Upload mixed audio
# 2. Click AI analysis (optional)
# 3. Click a suggested prompt OR type manually
# 4. Choose Keep/Remove mode
# 5. Click "Run Separation"
# 6. Verify: Output audio, residual, spectrograms, mask
```

### Test 4: Standalone Analyzer
```bash
# Test Gemini directly
cd src
python gemini_audio_analyzer.py ../data/ESC-50-master/ESC-50-master/audio/1-100032-A-0.wav

# Expected output:
# ✅ Analysis successful!
# 📊 Main Classes: ...
# 🔊 Specific Sounds: ...
# 📝 Characteristics: ...
# 💡 Separation Prompts: ...
```

## 🎯 Use Cases

### Before (Limited to ESC-50):
- ✅ Separate ESC-50 dataset files only
- ❌ Can't test your own audio
- ❌ Manual prompt guessing for unknown sounds
- ❌ No way to know what sounds are in a file

### After (Full Flexibility):
- ✅ Upload ANY audio file
- ✅ Mix multiple sounds for testing
- ✅ AI automatically detects all sounds
- ✅ Get AI-suggested prompts
- ✅ One-click prompt fill
- ✅ Test real-world audio scenarios

### Example Scenarios:

**1. Podcast Cleanup:**
- Upload: podcast.mp3 (voice + background music)
- AI detects: "male speech", "background music"
- Use prompt: "background music"
- Mode: Remove
- Result: Clean voice-only audio

**2. Field Recording:**
- Upload: nature_recording.wav (birds + wind + footsteps)
- AI detects: "bird chirping", "wind noise", "footsteps"
- Use prompt: "bird chirping"
- Mode: Keep
- Result: Isolated bird sounds

**3. Music Production:**
- Upload: full_mix.mp3 (vocals + drums + guitar + bass)
- AI detects: "singing voice", "drum beats", "electric guitar", "bass"
- Use prompt: "drum beats"
- Mode: Remove
- Result: Mix without drums

## 💡 Tips for Best Results

### AI Analysis:
- ✅ Works best with clear, distinct sounds
- ✅ Audio up to 5 minutes recommended
- ✅ Mix of 2-5 different sound types ideal
- ⚠️ Very noisy/distorted audio may get generic descriptions

### Separation Quality:
- ✅ Use AI-suggested prompts (they're optimized!)
- ✅ Be specific: "dog barking" > "animal"
- ✅ Try different k_components values (8-15 usually best)
- ✅ Adjust time range if sound appears in specific segment

## 🔍 Troubleshooting

### Upload Button Not Working
- **Check:** Browser console for errors
- **Fix:** Hard refresh (Ctrl+Shift+R)
- **Fix:** Restart server

### "API key not configured" Error
- **Check:** `echo $env:GEMINI_API_KEY`
- **Fix:** Set environment variable
- **Fix:** Add directly to `gemini_audio_analyzer.py`
- **See:** GEMINI_SETUP.md for detailed instructions

### Analysis Takes Forever
- **Cause:** Large file or slow internet
- **Fix:** Use shorter audio clips (< 5 min)
- **Fix:** Check internet connection
- **Note:** First request may take longer (model loading)

### Separation Not Working on Uploaded Files
- **Check:** File uploaded successfully?
- **Check:** File ID starts with `upload_`?
- **Check:** Server logs for errors
- **Fix:** Try re-uploading the file

## 📈 Future Enhancements

Possible additions:
- 🔄 Cache AI analysis results
- 📊 Show confidence scores for detected sounds
- 🎨 Visualize sound timeline
- 🔍 Multi-language support for prompts
- 💾 Save/load analysis history
- 🎵 Batch processing multiple files
- 🤝 Share analysis results

## 📝 Summary

**What You Can Now Do:**
1. ✅ Upload any audio file (not limited to dataset)
2. ✅ Let AI analyze and detect all sounds automatically
3. ✅ Get smart separation prompt suggestions
4. ✅ One-click to use suggested prompts
5. ✅ Separate uploaded audio with NMF+CLAP
6. ✅ Test real-world mixed audio scenarios

**Next Steps:**
1. 📖 Read GEMINI_SETUP.md
2. 🔑 Get your Gemini API key
3. ⚙️ Configure API key (environment variable)
4. 🧪 Test with your own audio files
5. 🎵 Enjoy AI-powered audio separation!

---

**Questions?** Check GEMINI_SETUP.md or the inline code comments!
