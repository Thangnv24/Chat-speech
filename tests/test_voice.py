"""
Real-time Speech-to-Text using Groq API (Whisper large-v3)
- Fast, accurate, free tier available
- Supports both file upload and real-time recording
- Includes audio preprocessing for optimal results
"""

import os
import io
import time
import wave
import numpy as np
import pyaudio
from pathlib import Path
from typing import Optional, Tuple
from groq import Groq
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import noisereduce as nr

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Get from .env
SAMPLE_RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
THRESHOLD = 500
SILENCE_LIMIT = 2.0  # seconds
MIN_RECORDING_LENGTH = 0.5  # seconds
MAX_RECORDING_LENGTH = 30  # seconds (Groq limit)

class AudioPreprocessor:
    """Preprocessing audio for optimal STT results"""
    
    @staticmethod
    def preprocess_audio(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """
        Apply preprocessing to improve transcription quality:
        1. Noise reduction
        2. Normalization
        3. Trim silence
        """
        # Noise reduction
        reduced_noise = nr.reduce_noise(
            y=audio_data, 
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.8
        )
        
        # Normalize volume
        normalized = reduced_noise / np.max(np.abs(reduced_noise))
        
        # Convert to int16
        processed = (normalized * 32767).astype(np.int16)
        
        return processed
    
    @staticmethod
    def trim_silence(audio_data: np.ndarray, threshold: int = 500) -> np.ndarray:
        """Remove leading and trailing silence"""
        # Find first and last non-silent samples
        non_silent = np.where(np.abs(audio_data) > threshold)[0]
        
        if len(non_silent) == 0:
            return audio_data
        
        start = max(0, non_silent[0] - int(SAMPLE_RATE * 0.1))  # Keep 0.1s before
        end = min(len(audio_data), non_silent[-1] + int(SAMPLE_RATE * 0.1))  # Keep 0.1s after
        
        return audio_data[start:end]
    
    @staticmethod
    def convert_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
        """Convert numpy array to WAV bytes"""
        byte_io = io.BytesIO()
        
        with wave.open(byte_io, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        byte_io.seek(0)
        return byte_io.read()
    
    @staticmethod
    def preprocess_file(file_path: str) -> bytes:
        """Preprocess audio file and return WAV bytes"""
        # Load audio file
        audio = AudioSegment.from_file(file_path)
        
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample to 16kHz
        if audio.frame_rate != SAMPLE_RATE:
            audio = audio.set_frame_rate(SAMPLE_RATE)
        
        # Normalize and compress
        audio = normalize(audio)
        audio = compress_dynamic_range(audio)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # Apply noise reduction
        processed = AudioPreprocessor.preprocess_audio(samples, SAMPLE_RATE)
        
        # Trim silence
        processed = AudioPreprocessor.trim_silence(processed)
        
        # Convert back to WAV bytes
        return AudioPreprocessor.convert_to_wav_bytes(processed)


class GroqSTT:
    """Speech-to-Text using Groq API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Get free API key from: https://console.groq.com"
            )
        
        self.client = Groq(api_key=self.api_key)
        self.preprocessor = AudioPreprocessor()
        print("✓ Groq STT initialized (Whisper large-v3)")
    
    def transcribe_file(self, file_path: str, language: str = "vi") -> dict:
        """
        Transcribe audio file
        
        Args:
            file_path: Path to audio file (wav, mp3, m4a, etc.)
            language: Language code (vi, en, etc.)
        
        Returns:
            dict with 'text', 'duration', 'processing_time'
        """
        print(f"📁 Processing file: {file_path}")
        start_time = time.time()
        
        # Preprocess audio
        print("🔧 Preprocessing audio...")
        processed_audio = self.preprocessor.preprocess_file(file_path)
        
        # Create temporary file-like object
        audio_file = io.BytesIO(processed_audio)
        audio_file.name = "audio.wav"
        
        # Transcribe
        print("🎤 Transcribing...")
        transcription = self.client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            language=language,
            response_format="verbose_json"
        )
        
        processing_time = time.time() - start_time
        
        result = {
            "text": transcription.text,
            "duration": transcription.duration,
            "processing_time": processing_time,
            "language": transcription.language
        }
        
        print(f"✓ Done in {processing_time:.2f}s")
        print(f"📝 Text: {result['text']}")
        
        return result
    
    def transcribe_realtime(self, language: str = "vi") -> dict:
        """
        Record and transcribe in real-time
        
        Args:
            language: Language code (vi, en, etc.)
        
        Returns:
            dict with 'text', 'duration', 'processing_time'
        """
        print("🎙️  Real-time recording started...")
        print(f"   Speak now (will stop after {SILENCE_LIMIT}s of silence)")
        
        # Record audio
        audio_data = self._record_audio()
        
        if audio_data is None:
            return {"text": "", "error": "No audio recorded"}
        
        # Preprocess
        print("🔧 Preprocessing audio...")
        processed = self.preprocessor.preprocess_audio(audio_data)
        processed = self.preprocessor.trim_silence(processed)
        
        # Check minimum length
        duration = len(processed) / SAMPLE_RATE
        if duration < MIN_RECORDING_LENGTH:
            print(f"⚠️  Recording too short ({duration:.1f}s)")
            return {"text": "", "error": "Recording too short"}
        
        # Convert to WAV bytes
        wav_bytes = self.preprocessor.convert_to_wav_bytes(processed)
        
        # Transcribe
        print("🎤 Transcribing...")
        start_time = time.time()
        
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"
        
        transcription = self.client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            language=language,
            response_format="verbose_json"
        )
        
        processing_time = time.time() - start_time
        
        result = {
            "text": transcription.text,
            "duration": duration,
            "processing_time": processing_time,
            "language": transcription.language
        }
        
        print(f"✓ Done in {processing_time:.2f}s")
        print(f"📝 Text: {result['text']}")
        
        return result
    
    def _record_audio(self) -> Optional[np.ndarray]:
        """Record audio from microphone with VAD"""
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            silent_chunks = 0
            is_speaking = False
            recording_start = time.time()
            
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Voice Activity Detection (VAD)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                volume = np.linalg.norm(audio_chunk)
                
                if volume > THRESHOLD:
                    silent_chunks = 0
                    if not is_speaking:
                        is_speaking = True
                        print("🔴 Speaking detected...")
                else:
                    if is_speaking:
                        silent_chunks += 1
                
                # Stop conditions
                recording_duration = time.time() - recording_start
                
                # Stop if silent for too long
                if is_speaking and silent_chunks > (SILENCE_LIMIT * SAMPLE_RATE / CHUNK):
                    print("⏸️  Silence detected, stopping...")
                    break
                
                # Stop if max length reached
                if recording_duration > MAX_RECORDING_LENGTH:
                    print(f"⏱️  Max recording length ({MAX_RECORDING_LENGTH}s) reached")
                    break
            
            stream.stop_stream()
            stream.close()
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None
        
        finally:
            p.terminate()


# ============= DEMO USAGE =============

def demo_file_transcription():
    """Demo: Transcribe audio file"""
    print("\n" + "="*60)
    print("DEMO 1: File Transcription")
    print("="*60 + "\n")
    
    stt = GroqSTT()
    
    # Example file path
    file_path = "voice.wav"
    
    if not Path(file_path).exists():
        print(f"⚠️  File not found: {file_path}")
        print("   Please provide a valid audio file path")
        return
    
    result = stt.transcribe_file(file_path, language="en")
    
    print("\n" + "-"*60)
    print(f"Text: {result['text']}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Processing: {result['processing_time']:.2f}s")
    print(f"Speed: {result['duration']/result['processing_time']:.1f}x real-time")
    print("-"*60)


def demo_realtime_recording():
    """Demo: Real-time recording and transcription"""
    print("\n" + "="*60)
    print("DEMO 2: Real-time Recording")
    print("="*60 + "\n")
    
    stt = GroqSTT()
    
    try:
        while True:
            print("\n[Press Enter to start recording, Ctrl+C to exit]")
            input()
            
            result = stt.transcribe_realtime(language="vi")
            
            if result.get("error"):
                print(f"⚠️  {result['error']}")
                continue
            
            print("\n" + "-"*60)
            print(f"Text: {result['text']}")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"Processing: {result['processing_time']:.2f}s")
            print("-"*60)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    # Check API key
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in environment variables")
        print("   Get free API key from: https://console.groq.com")
        print("   Add to .env file: GROQ_API_KEY=your_key_here")
        exit(1)
    
    # Run demos
    print("\n🎤 Speech-to-Text Demo (Groq + Whisper large-v3)")
    print("\nChoose demo:")
    print("1. Transcribe audio file")
    print("2. Real-time recording")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        demo_file_transcription()
    elif choice == "2":
        demo_realtime_recording()
    elif choice == "3":
        demo_file_transcription()
        demo_realtime_recording()
    else:
        print("Invalid choice")