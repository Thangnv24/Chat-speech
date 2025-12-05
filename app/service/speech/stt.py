import os
import sys
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
from app.utils.logger import setup_logging

# Initialize logger
logger = setup_logging("stt")


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  
SAMPLE_RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
THRESHOLD = 1000
SILENCE_LIMIT = 2.0  # seconds
MIN_RECORDING_LENGTH = 0.5  # seconds
MAX_RECORDING_LENGTH = 30  # seconds (Groq limit)

class AudioPreprocessor:
    @staticmethod
    def preprocess_audio(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
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
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found"
            )
        
        self.client = Groq(api_key=self.api_key)
        self.preprocessor = AudioPreprocessor()
        logger.info("Groq STT initialized whisper")
    
    def transcribe_file(self, file_path: str, language: str = "vi") -> dict:
        start_time = time.time()
        
        # Preprocess audio
        logger.info("Preprocessing audio...")
        processed_audio = self.preprocessor.preprocess_file(file_path)
        
        # Create temporary file-like object
        audio_file = io.BytesIO(processed_audio)
        audio_file.name = "audio.wav"
        
        # Transcribe
        transcription = self.client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
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
        
        logger.info(f"Done in {processing_time:.2f}s")
        logger.info(f"Text: {result['text']}")        
        return result
    
    def transcribe_realtime(self, language: str = "vi") -> dict:
        
        logger.info(f"Speak now (will stop after {SILENCE_LIMIT}s of silence)")
        
        # Record audio
        audio_data = self._record_audio()
        
        if audio_data is None:
            return {"text": "", "error": "No audio recorded"}
        
        # Preprocess
        logger.info("Preprocessing audio...")
        processed = self.preprocessor.preprocess_audio(audio_data)
        processed = self.preprocessor.trim_silence(processed)
        
        # Check minimum length
        duration = len(processed) / SAMPLE_RATE
        if duration < MIN_RECORDING_LENGTH:
            logger.error(f"Recording too short ({duration:.1f}s)")
            return {"text": "", "error": "Recording too short"}
        
        # Convert to WAV bytes
        wav_bytes = self.preprocessor.convert_to_wav_bytes(processed)
        
        # Transcribe
        start_time = time.time()
        
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"
        
        transcription = self.client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
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
        
        logger.info(f"Done in {processing_time:.2f}s")
        logger.info(f"Text: {result['text']}")
        
        return result
    
    def _record_audio(self) -> Optional[np.ndarray]:
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
            timeout = 5.0
            
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
                        logger.info("Speaking detected...")
                else:
                    if is_speaking:
                        silent_chunks += 1
                
                # Stop conditions
                recording_duration = time.time() - recording_start
                
                # Stop if No speech deetected too long
                if not is_speaking and recording_duration > timeout:
                    logger.warning("No speech detected")
                    return None

                # Stop if silent for too long
                if is_speaking and silent_chunks > (SILENCE_LIMIT * SAMPLE_RATE / CHUNK):
                    logger.warning("Silence detected, stopping...")
                    break
                
                # Stop if max length reached
                if recording_duration > MAX_RECORDING_LENGTH:
                    logger.warning(f"Max recording length ({MAX_RECORDING_LENGTH}s) reached")
                    break
            
            stream.stop_stream()
            stream.close()
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None
        
        finally:
            p.terminate()


def demo_file_transcription(file_path):
    stt = GroqSTT()
    # file_path = "data/t1.wav"
    
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return
    
    result = stt.transcribe_file(file_path, language="en")
    
    logger.info(f"Text: {result['text']}")
    logger.info(f"Duration: {result['duration']:.2f}s")
    logger.info(f"Processing: {result['processing_time']:.2f}s")
    logger.info(f"Speed: {result['duration']/result['processing_time']:.1f}x real-time")

    return result['text']

def demo_realtime_recording():
    stt = GroqSTT()
    
    try:
        while True:
            print("Press Enter to start recording, Ctrl+C to exit")
            input()
            
            result = stt.transcribe_realtime(language="vi")
            
            if result.get("error"):
                logger.error(f"Error: {result['error']}")
                continue
            
        logger.info(f"Text: {result['text']}")
        logger.info(f"Duration: {result['duration']:.2f}s")
        logger.info(f"Processing: {result['processing_time']:.2f}s")
            
    except KeyboardInterrupt:
        print("End")


if __name__ == "__main__":
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not found")
        exit(1)

    print("\nSpeech-to-Text Demo (Groq + Whisper large-v3)")
    print("\nChoose demo:")
    print("1. Transcribe audio file")
    print("2. Real-time recording")
    print("3. Both")

    choice = input("\nEnter choice (1/2/3): ").strip()

    file_path = "data/pyta.wav"

    if choice == "1":
        text = demo_file_transcription(file_path)
        print(text)
    elif choice == "2":
        demo_realtime_recording()
    elif choice == "3":
        demo_file_transcription(file_path)
        demo_realtime_recording()
    else:
        print("Invalid choice")