# 🎤 Voice-to-Text Setup Guide

## Tổng Quan

Giải pháp Speech-to-Text mới sử dụng **Groq API** với Whisper large-v3:
- ✅ **Miễn phí**: Free tier hào phóng
- ✅ **Cực nhanh**: ~10-20x nhanh hơn real-time
- ✅ **Chính xác cao**: Whisper large-v3 model
- ✅ **Không cần GPU**: Chạy trên cloud
- ✅ **Hỗ trợ tiếng Việt**: Native Vietnamese support

## Cài Đặt

### 1. Cài Dependencies

```bash
pip install -r requirements-voice.txt
```

**Lưu ý cho Windows:**
- PyAudio cần cài riêng: `pip install pipwin && pipwin install pyaudio`
- Hoặc download wheel từ: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

**Lưu ý cho Linux:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install -r requirements-voice.txt
```

**Lưu ý cho macOS:**
```bash
brew install portaudio
pip install -r requirements-voice.txt
```

### 2. Lấy Groq API Key (Miễn Phí)

1. Truy cập: https://console.groq.com
2. Đăng ký tài khoản (miễn phí)
3. Tạo API key tại: https://console.groq.com/keys
4. Copy API key

### 3. Cấu Hình

Thêm vào file `.env`:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

## Sử Dụng

### Chạy Demo

```bash
python tests/test_voice.py
```

Chọn chế độ:
- **1**: Transcribe file audio có sẵn
- **2**: Thu âm real-time từ microphone
- **3**: Chạy cả hai

### Sử Dụng Trong Code

#### 1. Transcribe File Audio

```python
from tests.test_voice import GroqSTT

stt = GroqSTT()
result = stt.transcribe_file("audio.wav", language="vi")

print(result['text'])
print(f"Duration: {result['duration']:.2f}s")
print(f"Processing: {result['processing_time']:.2f}s")
```

#### 2. Real-time Recording

```python
from tests.test_voice import GroqSTT

stt = GroqSTT()
result = stt.transcribe_realtime(language="vi")

print(result['text'])
```

## Tính Năng

### Audio Preprocessing

Code tự động xử lý audio để tối ưu kết quả:

1. **Noise Reduction**: Giảm nhiễu nền
2. **Normalization**: Chuẩn hóa âm lượng
3. **Silence Trimming**: Cắt bỏ khoảng lặng đầu/cuối
4. **Resampling**: Chuyển về 16kHz (tối ưu cho Whisper)
5. **Mono Conversion**: Chuyển về 1 channel

### Voice Activity Detection (VAD)

- Tự động phát hiện khi bạn bắt đầu nói
- Dừng sau 2 giây im lặng
- Giới hạn tối đa 30 giây (Groq limit)

### Supported Formats

- **Input**: WAV, MP3, M4A, FLAC, OGG, và hầu hết các format phổ biến
- **Output**: Text transcription với metadata

## Performance

### Groq API (Whisper large-v3)

- **Tốc độ**: ~10-20x faster than real-time
- **Ví dụ**: Audio 10s → Transcribe trong ~0.5-1s
- **Độ chính xác**: 95%+ cho tiếng Việt rõ ràng
- **Free tier**: 
  - 14,400 requests/day
  - ~6 hours audio/day

### So Sánh với Local Whisper

| Metric | Local (tiny) | Local (base) | Groq API (large-v3) |
|--------|-------------|--------------|---------------------|
| Tốc độ | 2-3x RT | 1-2x RT | 10-20x RT |
| Độ chính xác | 70-80% | 80-85% | 95%+ |
| RAM | 1GB | 1.5GB | 0MB (cloud) |
| Setup | Phức tạp | Phức tạp | Đơn giản |

*RT = Real-time

## Troubleshooting

### Lỗi: "No module named 'pyaudio'"

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

### Lỗi: "GROQ_API_KEY not found"

Kiểm tra:
1. File `.env` có chứa `GROQ_API_KEY=...`
2. API key đúng format (bắt đầu với `gsk_`)
3. Restart terminal sau khi thêm vào `.env`

### Lỗi: "No default input device"

Microphone không được phát hiện:
1. Kiểm tra microphone đã kết nối
2. Cấp quyền microphone cho terminal/IDE
3. Test với: `python -m pyaudio`

### Audio Quality Kém

Cải thiện:
1. Nói rõ ràng, không quá nhanh
2. Giảm nhiễu nền (tắt quạt, đóng cửa)
3. Microphone gần miệng hơn (15-30cm)
4. Tăng `THRESHOLD` nếu bắt quá nhiều nhiễu

## Advanced Usage

### Custom Preprocessing

```python
from tests.test_voice import AudioPreprocessor

preprocessor = AudioPreprocessor()

# Preprocess file
wav_bytes = preprocessor.preprocess_file("audio.mp3")

# Preprocess numpy array
import numpy as np
audio_data = np.array([...])  # Your audio data
processed = preprocessor.preprocess_audio(audio_data, sample_rate=16000)
```

### Adjust VAD Sensitivity

Trong `test_voice.py`, điều chỉnh:

```python
THRESHOLD = 500  # Tăng nếu bắt nhiễu, giảm nếu không bắt giọng nói
SILENCE_LIMIT = 2.0  # Thời gian im lặng trước khi dừng
```

### Multi-language Support

```python
# English
result = stt.transcribe_file("audio.wav", language="en")

# Auto-detect
result = stt.transcribe_file("audio.wav", language=None)
```

## API Limits

### Groq Free Tier

- **Requests**: 14,400/day (~600/hour)
- **Rate limit**: 30 requests/minute
- **Audio length**: Max 25MB per file
- **No credit card required**

Đủ cho hầu hết use cases cá nhân và testing.

## Next Steps

1. ✅ Tích hợp vào FastAPI endpoint
2. ✅ Thêm WebSocket cho streaming
3. ✅ Cache results để tiết kiệm API calls
4. ✅ Thêm translation (STT → Translation)
5. ✅ Kết hợp với RAG system

## Resources

- Groq Console: https://console.groq.com
- Groq Docs: https://console.groq.com/docs
- Whisper Paper: https://arxiv.org/abs/2212.04356
- PyAudio Docs: https://people.csail.mit.edu/hubert/pyaudio/
