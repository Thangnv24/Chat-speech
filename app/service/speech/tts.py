import os
import re
from groq import Groq
from pathlib import Path
from langdetect import detect, LangDetectException
from app.utils.logger import setup_logging, logger
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import wave 

logger = setup_logging("tts")
ELEVEN_API_KEY=os.getenv("ELEVEN_API_KEY", "")

client = ElevenLabs(
    api_key=ELEVEN_API_KEY
)
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "") 
# client = Groq(api_key=GROQ_API_KEY)


MATH_PHIL_SYMBOLS = {
    "vi": {
        # Math
        "+": " cộng ",
        "(": " mở ngoặc ",
        ")": " đóng ngoặc ",
        "=": " bằng",
        ">": " lớn hơn",
        "<": " nhỏ hơn",
        "∀": " với mọi ",
        "∃": " tồn tại ",
        "∄": " không tồn tại ",
        "∈": " thuộc ",
        "∉": " không thuộc ",
        "⊂": " là con của ",
        "→": " suy ra ",
        "⇒": " kéo theo ",
        "⇔": " tương đương với ",
        "∑": " tổng của ",
        "∫": " tích phân ",
        "∴": " vì vậy, ",
        "∵": " bởi vì, ",
        "≠": " khác ",
        "≈": " xấp xỉ ",
        "≤": " nhỏ hơn hoặc bằng ",
        "≥": " lớn hơn hoặc bằng ",
        "∞": " vô cực ",
        # Philo
        "§": " điều ",
        "†": " chú thích ",
        "etc.": " vân vân ",
    },
    "en": {
        "+": " plus ",
        "(": " open ",
        ")": " close ",
        "∀": " for all ",
        "∃": " there exists ",
        "∄": " there does not exist ",
        "∈": " in ",
        "∉": " not in ",
        "⊂": " is a subset of ",
        "→": " implies ",
        "⇒": " implies ",
        "⇔": " is equivalent to ",
        "∑": " sum of ",
        "∫": " integral of ",
        "∴": " therefore, ",
        "∵": " because, ",
        "≠": " not equal to ",
        "≈": " approximately ",
        "≤": " less than or equal to ",
        "≥": " greater than or equal to ",
        "∞": " infinity ",
        ">": " greater than",
        "<": " less than",
        "=": " equal",
        # philo
        "§": " section ",
        "†": " dagger ",
        "etc.": " et cetera ",
    }
}

def detect_language(text):
    try:
        lang = detect(text)
        if lang in ['vi', 'en']:
            return lang
        return 'en' 
    except LangDetectException:
        return 'en'

# Normalize text for reading in en and vi
def normalize_text_for_reading(text, lang='en'):
    symbols = MATH_PHIL_SYMBOLS.get(lang, MATH_PHIL_SYMBOLS['en'])
    for symbol, replacement in symbols.items():
        text = text.replace(symbol, replacement)

    if lang == 'vi':
        text = re.sub(r'(\w+)\^2', r'\1 bình phương', text)
        text = re.sub(r'(\w+)\^3', r'\1 lập phương', text)
    else:
        text = re.sub(r'(\w+)\^2', r'\1 squared', text)
        text = re.sub(r'(\w+)\^3', r'\1 cubed', text)

    # Escape []" 
    text = re.sub(r'\[\d+\]', '', text) 

    text = text.replace(';', ',')
    text = text.replace('—', ',')
    
    # Normalize space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def text_to_speech(input_text, output_file="speech.wav"):
    lang = detect_language(input_text)
    logger.info(f"Detected language: {lang}")

    processed_text = normalize_text_for_reading(input_text, lang)
    logger.info(f"Normalized text: {processed_text}")

    # if lang == 'vi':
    #     # Dùng SSML tag <lang> để chỉ định ngôn ngữ phát âm là tiếng Việt
    #     tts_input_text = f'<speak><lang xml:lang="vi-VN">{processed_text}</lang></speak>'
    # else:
    #     tts_input_text = processed_text

    speech_file_path = Path(__file__).parent / output_file
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    # Model
    try:
        if lang == 'vi':
            # Dùng SSML tag <lang> để chỉ định ngôn ngữ phát âm là tiếng Việt
            tts_input_text = f'<speak><lang xml:lang="vi-VN">{processed_text}</lang></speak>'
            response = client.text_to_speech.convert(
                text=tts_input_text,
                voice_id="1d5Bb0SMBPB10Gx6iQeu",
                model_id="eleven_turbo_v2_5",
                output_format="pcm_16000",
                voice_settings={
                    "stability": 0.5,           
                    "similarity_boost": 0.75,   
                }
            )
        else:
            tts_input_text = processed_text
            response = client.text_to_speech.convert(
                text=tts_input_text,
                voice_id="FxZjRiAEBESrb7srpme7",
                model_id="eleven_multilingual_v2",
                output_format="pcm_16000",
                voice_settings={
                    "stability": 0.5,           
                    "similarity_boost": 0.75,   
                }
            )
        
        # Thu thập dữ liệu từ Generator/Iterator
        audio_bytes = b''.join(response)
        # Save
        with wave.open(str(speech_file_path), "wb") as f:
            f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            
            f.writeframes(audio_bytes) 

        logger.info(f"Audio saved to: {speech_file_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")


# text_en = "Therefore, ∀x ∈ R, if x^2 > 0 → x ≠ 0. As stated in [1], this is trivial."
# text_to_speech(text_en, "speech_en.wav")

# text_vi = "Vì vậy, ∀x ∈ R, nếu x^2 > 0 → x ≠ 0. Triết học hiện sinh cho rằng a = b"
# text_vi =  "Xin chào các bạn, lại là Chao đây. Hôm nay Chao sẽ đưa các bạn đi tham quan bảo tàng lịch sử quốc gia nga nhé"
# text_to_speech(text_vi, "speech_vi.wav")
