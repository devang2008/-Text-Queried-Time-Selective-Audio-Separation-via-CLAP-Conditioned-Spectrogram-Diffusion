"""
Gemini API integration for audio content analysis.
Uses Google's Gemini 1.5 models to detect sounds in uploaded audio files.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict, Optional

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API - reads from .env file or environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def configure_gemini():
    """Configure Gemini API with your API key."""
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    return False

def analyze_audio_with_gemini(audio_path: str) -> Dict[str, any]:
    """
    Analyze audio file using Gemini API to detect sound classes.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with detected sounds, descriptions, and suggestions
    """
    if not configure_gemini():
        return {
            "error": "Gemini API key not configured",
            "message": "Please set GEMINI_API_KEY environment variable or add it to gemini_audio_analyzer.py"
        }
    
    try:
        # Upload audio file to Gemini
        audio_file = genai.upload_file(path=audio_path)
        
        # Use Gemini 2.5 Flash for audio analysis
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Prompt for audio analysis - requesting short labels only
        prompt = """Analyze this audio file and identify the sounds present.

List only the specific sounds you can hear, using VERY SHORT labels (maximum 2 words each).

Examples of good labels:
- Speech
- Music
- Applause
- Dog barking
- Car horn
- Rain
- Footsteps
- Piano

Format your response as:
SPECIFIC SOUNDS: [comma-separated list of 1-2 word sound labels]
"""

        # Generate response
        response = model.generate_content([prompt, audio_file])
        
        # Parse the response
        result = parse_gemini_response(response.text)
        result["raw_response"] = response.text
        result["success"] = True
        
        # Clean up uploaded file
        genai.delete_file(audio_file.name)
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False,
            "message": f"Failed to analyze audio: {str(e)}"
        }

def parse_gemini_response(response_text: str) -> Dict[str, any]:
    """
    Parse Gemini's structured response into a dictionary.
    
    Args:
        response_text: Raw response from Gemini
        
    Returns:
        Parsed dictionary with sound information
    """
    result = {
        "main_classes": [],
        "specific_sounds": [],
        "characteristics": "",
        "separation_prompts": []
    }
    
    lines = response_text.strip().split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect sections
        if line.startswith("MAIN CLASSES:"):
            current_section = "main_classes"
            content = line.replace("MAIN CLASSES:", "").strip()
            if content:
                result["main_classes"] = [s.strip() for s in content.split(',')]
        elif line.startswith("SPECIFIC SOUNDS:"):
            current_section = "specific_sounds"
            content = line.replace("SPECIFIC SOUNDS:", "").strip()
            if content:
                result["specific_sounds"] = [s.strip() for s in content.split(',')]
        elif line.startswith("CHARACTERISTICS:"):
            current_section = "characteristics"
            content = line.replace("CHARACTERISTICS:", "").strip()
            result["characteristics"] = content
        elif line.startswith("SEPARATION PROMPTS:"):
            current_section = "separation_prompts"
            content = line.replace("SEPARATION PROMPTS:", "").strip()
            if content:
                result["separation_prompts"].append(content)
        else:
            # Continue adding to current section
            if current_section == "characteristics":
                result["characteristics"] += " " + line
            elif current_section == "separation_prompts":
                # Extract prompts (often numbered or bulleted)
                clean_line = line.lstrip('0123456789.-•*) ').strip()
                if clean_line and len(clean_line) > 3:
                    result["separation_prompts"].append(clean_line)
    
    # Clean up
    result["characteristics"] = result["characteristics"].strip()
    
    # If parsing failed, try to extract any useful information
    if not result["specific_sounds"] and not result["main_classes"]:
        # Fallback: just return the full response as characteristics
        result["characteristics"] = response_text
        result["note"] = "Could not parse structured response, showing raw analysis"
    
    return result

def get_quick_sound_suggestions(audio_path: str, max_suggestions: int = 5) -> List[str]:
    """
    Get quick sound detection suggestions for separation prompts.
    
    Args:
        audio_path: Path to audio file
        max_suggestions: Maximum number of suggestions to return
        
    Returns:
        List of suggested text prompts for separation
    """
    result = analyze_audio_with_gemini(audio_path)
    
    if result.get("success"):
        return result.get("separation_prompts", [])[:max_suggestions]
    else:
        return []

# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gemini_audio_analyzer.py <audio_file_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    print("Analyzing audio with Gemini API...")
    result = analyze_audio_with_gemini(audio_path)
    
    if result.get("success"):
        print("\n✅ Analysis successful!")
        print(f"\n📊 Main Classes: {', '.join(result['main_classes'])}")
        print(f"\n🔊 Specific Sounds: {', '.join(result['specific_sounds'])}")
        print(f"\n📝 Characteristics: {result['characteristics']}")
        print(f"\n💡 Separation Prompts:")
        for i, prompt in enumerate(result['separation_prompts'], 1):
            print(f"  {i}. {prompt}")
    else:
        print(f"\n❌ Error: {result.get('message', result.get('error'))}")
