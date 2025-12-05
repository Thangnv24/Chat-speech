from stt import demo_file_transcription
from tts import text_to_speech
import os
from google import genai
from pathlib import Path
from app.utils.logger import setup_logging, logger

logger = setup_logging("Speech-to-speech")
gemini_api_key = os.getenv("GEMINI_API_KEY", "")

try:
    client_gemini = genai.Client()
except Exception:
    logger.error("Gemini key not found")
    exit()

def run_speech_to_speech_process(audio_input_file, audio_output_file):
    user_prompt = demo_file_transcription(audio_input_file)
    # user_prompt = "cho tôi công thức (a+b)^2 "
    if not user_prompt:
        logger.warning("Can not detect voice")
        return

    try:
        response = client_gemini.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config={
                "system_instruction": "Bạn là một trợ lý thông minh và súc tích. Hãy luôn trả lời cùng ngôn ngữ với câu hỏi, chỉ sử dụng tối đa hai (2) câu ngắn gọn cho mỗi câu hỏi. Tránh các lời mở đầu hoặc kết thúc dài dòng.  **Không sử dụng bất kỳ ký hiệu định dạng công thức nào (như $ hay LaTeX) trong câu trả lời, hãy viết công thức ra bằng chữ thuần túy.**",
                "temperature": 0.5
            }
        )
        llm_response_text = response.text
        logger.info(f"Gemini answers: {llm_response_text[:80]}...")
        
    except Exception as e:
        logger.error(f"Error {e}")
        return

    # Text to speech
    text_to_speech(llm_response_text, audio_output_file)
    
INPUT_WAV = "data/pyta.wav" 
OUTPUT_WAV = "gemini_response.wav"

Path(INPUT_WAV).touch() 

run_speech_to_speech_process(INPUT_WAV, OUTPUT_WAV)