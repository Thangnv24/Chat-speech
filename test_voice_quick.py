"""
Quick test script for voice transcription
Run this to verify your setup works
"""

import os
from pathlib import Path

# Check if GROQ_API_KEY is set
api_key = os.getenv("GROQ_API_KEY")
print(api_key)
if not api_key:
    print("❌ GROQ_API_KEY not found!")
    print("\n📝 Setup instructions:")
    print("1. Get free API key from: https://console.groq.com")
    print("2. Add to .env file: GROQ_API_KEY=your_key_here")
    print("3. Restart terminal and run again")
    exit(1)

print("✓ GROQ_API_KEY found")

# Check dependencies
try:
    import groq
    print("✓ groq installed")
except ImportError:
    print("❌ groq not installed")
    print("   Run: pip install groq")
    exit(1)

try:
    import pyaudio
    print("✓ pyaudio installed")
except ImportError:
    print("⚠️  pyaudio not installed (needed for real-time recording)")
    print("   Windows: pip install pipwin && pipwin install pyaudio")
    print("   Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
    print("   macOS: brew install portaudio && pip install pyaudio")

try:
    import pydub
    print("✓ pydub installed")
except ImportError:
    print("⚠️  pydub not installed (needed for audio preprocessing)")
    print("   Run: pip install pydub")

try:
    import noisereduce
    print("✓ noisereduce installed")
except ImportError:
    print("⚠️  noisereduce not installed (needed for noise reduction)")
    print("   Run: pip install noisereduce")

print("\n" + "="*60)
print("🎤 Testing Groq API connection...")
print("="*60)

try:
    from groq import Groq
    client = Groq(api_key=api_key)
    
    # Test with a simple text completion (free)
    print("\n✓ Groq API connection successful!")
    print("✓ Ready to transcribe audio")
    
    print("\n" + "="*60)
    print("🚀 Next steps:")
    print("="*60)
    print("\n1. Run full demo:")
    print("   python tests/test_voice.py")
    print("\n2. Or test with your audio file:")
    print("   from tests.test_voice import GroqSTT")
    print("   stt = GroqSTT()")
    print("   result = stt.transcribe_file('your_audio.wav')")
    print("\n3. Or start FastAPI server:")
    print("   uvicorn app.main:app --reload")
    print("   Then upload audio at: http://localhost:8000/docs")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPlease check your API key is correct")
