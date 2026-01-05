# 🎤 Voice Chat Guide - Speech-to-Speech RAG

## 📋 Tổng quan

Voice Chat cho phép bạn trò chuyện với AI bằng giọng nói:
1. **Nói** câu hỏi của bạn
2. **AI transcribe** giọng nói thành text (STT)
3. **RAG** tìm kiếm thông tin trong vector store
4. **AI trả lời** bằng text
5. **TTS** (tùy chọn) đọc câu trả lời

---

## 🔧 Yêu cầu

### 1. API Keys

Cần có trong file `.env`:

```env
# Speech-to-Text (Groq Whisper)
GROQ_API_KEY=your_groq_api_key

# Text-to-Speech (ElevenLabs)
ELEVEN_API_KEY=your_elevenlabs_api_key

# LLM cho RAG (chọn 1)
GEMINI_API_KEY=your_gemini_key
# hoặc
QWEN_API_KEY=your_qwen_key
# hoặc
OLLAMA_BASE_URL=http://localhost:11434

# Qdrant
QDRANT_URL=http://localhost:6333
```

### 2. Python Dependencies

```bash
pip install groq elevenlabs langdetect noisereduce pyaudio pydub
```

### 3. System Dependencies

#### Windows:
- PyAudio: `pip install pipwin && pipwin install pyaudio`
- FFmpeg: Download từ https://ffmpeg.org/download.html

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg
pip install pyaudio
```

#### Mac:
```bash
brew install portaudio ffmpeg
pip install pyaudio
```

---

## 🚀 Cách sử dụng

### 1. Khởi động ứng dụng

```bash
# Đảm bảo Qdrant và PostgreSQL đang chạy
docker-compose up -d db qdrant

# Đảm bảo đã ingest dữ liệu
python ingest_data.py

# Chạy app
uvicorn app.main:app --reload
```

### 2. Truy cập Voice Chat

Mở browser: http://localhost:8000/static/voice_chat.html

### 3. Tạo session mới

Click nút **"+ New Chat"** trong sidebar

### 4. Bắt đầu voice chat

#### Cách 1: Click nút microphone 🎤
1. Click nút microphone màu đỏ
2. Nói câu hỏi của bạn
3. Click lại để dừng recording
4. Đợi AI xử lý và trả lời

#### Cách 2: Dùng phím tắt
1. Nhấn **Space** để bắt đầu recording
2. Nói câu hỏi
3. Nhấn **Space** lại để dừng

### 5. Chế độ Text (tùy chọn)

Click nút **"💬 Text"** để chuyển sang text chat

---

## 🎯 Tính năng

### ✅ Voice Chat
- ✅ **Speech-to-Text**: Groq Whisper large-v3-turbo
- ✅ **RAG Integration**: Tìm kiếm trong vector store
- ✅ **Auto-save**: Lưu vào database như text chat
- ✅ **Multi-language**: Hỗ trợ tiếng Việt và tiếng Anh
- ✅ **TTS Response**: Tự động phát âm câu trả lời (tùy chọn)

### ✅ Session Management
- ✅ **Create sessions**: Tạo session mới
- ✅ **Rename sessions**: Đổi tên session (click ✏️)
- ✅ **Switch sessions**: Chuyển đổi giữa các session
- ✅ **Message history**: Xem lịch sử chat

### ✅ UI Features
- ✅ **Recording indicator**: Hiển thị khi đang recording
- ✅ **Typing indicator**: Hiển thị khi AI đang xử lý
- ✅ **Mode toggle**: Chuyển đổi Text/Voice
- ✅ **Keyboard shortcuts**: Space để record, Enter để gửi text
- ✅ **Responsive design**: Hoạt động trên mobile

---

## 📡 API Endpoints

### Voice Chat

```bash
# Voice chat (upload audio file)
POST /api/v1/voice/chat
Content-Type: multipart/form-data

Parameters:
- session_id: UUID (query param)
- audio_file: File (form data)
- k: int (optional, default=5)
- search_mode: str (optional, default="hybrid")
- language: str (optional, default="vi")

Response:
{
  "transcribed_text": "câu hỏi đã transcribe",
  "answer": "câu trả lời từ RAG",
  "audio_duration": 3.5,
  "processing_time": 2.1,
  "query_time": 0.8,
  "num_retrieved": 5,
  "user_message": {...},
  "ai_message": {...}
}
```

### Text-to-Speech

```bash
# Convert text to speech
POST /api/v1/voice/tts?text=Hello&language=auto

Response: audio/wav file
```

### Health Check

```bash
# Check voice service health
GET /api/v1/voice/health

Response:
{
  "status": "healthy",
  "components": {
    "stt": "configured",
    "tts": "configured",
    "rag": "healthy"
  }
}
```

### Session Rename

```bash
# Rename session
PATCH /api/v1/sessions/{session_id}/name?session_name=New%20Name

Response:
{
  "session_id": "...",
  "session_name": "New Name",
  "user_id": "...",
  "started_at": "..."
}
```

---

## 🧪 Testing

### Test Voice Service

```bash
# Check health
curl http://localhost:8000/api/v1/voice/health

# Test TTS
curl -X POST "http://localhost:8000/api/v1/voice/tts?text=Hello%20World&language=en" \
  --output test_speech.wav

# Play audio (Linux)
aplay test_speech.wav

# Play audio (Mac)
afplay test_speech.wav

# Play audio (Windows)
start test_speech.wav
```

### Test Voice Chat với curl

```bash
# Record audio first (use any tool)
# Then upload:

curl -X POST "http://localhost:8000/api/v1/voice/chat?session_id=YOUR_SESSION_ID&language=vi" \
  -F "audio_file=@voice_input.wav" \
  -H "Content-Type: multipart/form-data"
```

### Test STT trực tiếp

```bash
# Test STT với file audio
python -c "
from app.service.speech.stt import GroqSTT
stt = GroqSTT()
result = stt.transcribe_file('data/test_audio.wav', language='vi')
print(result)
"
```

### Test TTS trực tiếp

```bash
# Test TTS
python -c "
from app.service.speech.tts import text_to_speech
text_to_speech('Xin chào, tôi là AI assistant', 'test_output.wav')
print('Audio saved to test_output.wav')
"
```

---

## 🎨 UI Customization

### Thay đổi màu sắc

Sửa trong `static/voice_chat.html`:

```css
/* Recording button color */
#voice-btn {
    background: #e74c3c;  /* Đỏ */
}

#voice-btn.recording {
    background: #27ae60;  /* Xanh lá khi recording */
}

/* Mode toggle */
.mode-btn.active {
    background: #3498db;  /* Xanh dương */
}
```

### Thay đổi ngôn ngữ mặc định

Sửa trong `static/voice_chat.html`:

```javascript
// Đổi từ 'vi' sang 'en'
const res = await fetch(`${API}/voice/chat?session_id=${currentSession}&language=en&...`);
```

---

## 🐛 Troubleshooting

### Lỗi: "Microphone access denied"

**Nguyên nhân**: Browser chưa được cấp quyền microphone

**Giải pháp**:
1. Click vào icon 🔒 trên address bar
2. Cho phép microphone access
3. Reload page

### Lỗi: "GROQ_API_KEY not found"

**Giải pháp**:
```bash
# Thêm vào .env
GROQ_API_KEY=your_groq_api_key

# Restart app
uvicorn app.main:app --reload
```

### Lỗi: "Could not transcribe audio"

**Nguyên nhân**: Audio quality thấp hoặc không có giọng nói

**Giải pháp**:
- Nói rõ ràng hơn
- Kiểm tra microphone
- Giảm noise xung quanh
- Thử lại với audio file test

### Lỗi: "Vector store not initialized"

**Giải pháp**:
```bash
# Ingest dữ liệu trước
python ingest_data.py

# Restart app
uvicorn app.main:app --reload
```

### Audio không phát (TTS)

**Nguyên nhân**: Browser block autoplay

**Giải pháp**:
1. Click vào page trước
2. Hoặc tắt TTS autoplay trong code:
```javascript
// Comment dòng này trong voice_chat.html
// await playTTS(data.answer);
```

### Recording quá ngắn

**Giải pháp**: Nói lâu hơn (tối thiểu 0.5 giây)

### Recording quá dài

**Giải pháp**: Groq giới hạn 30 giây, hãy nói ngắn gọn hơn

---

## 📊 Performance Tips

### 1. Tối ưu STT
- Sử dụng audio quality tốt (16kHz, mono)
- Giảm background noise
- Nói rõ ràng

### 2. Tối ưu RAG
- Giảm `k` (số documents retrieve) nếu chậm
- Dùng `search_mode="dense"` thay vì `"hybrid"` nếu muốn nhanh hơn

### 3. Tối ưu TTS
- Tắt TTS autoplay nếu không cần
- Cache audio responses

---

## 🔐 Security Notes

### API Keys
- **KHÔNG** commit API keys vào Git
- Dùng `.env` file (đã có trong `.gitignore`)
- Rotate keys định kỳ

### Microphone Access
- Chỉ request khi cần
- Giải phóng stream sau khi dùng xong

### File Upload
- Giới hạn file size (hiện tại: 30s audio)
- Validate file type
- Clean up temp files

---

## 🎯 Next Steps

### Tính năng có thể thêm:

1. **Real-time streaming STT**
   - Transcribe trong khi nói
   - Hiển thị text real-time

2. **Voice activity detection (VAD)**
   - Tự động dừng khi im lặng
   - Không cần click stop

3. **Multi-turn conversation**
   - Context awareness
   - Follow-up questions

4. **Voice profiles**
   - Lưu voice settings
   - Custom TTS voices

5. **Audio visualization**
   - Waveform display
   - Volume meter

6. **Offline mode**
   - Local STT/TTS models
   - No API keys needed

---

## 📚 Resources

### API Documentation
- Groq Whisper: https://console.groq.com/docs/speech-text
- ElevenLabs: https://elevenlabs.io/docs/api-reference/text-to-speech
- Qdrant: https://qdrant.tech/documentation/

### Libraries
- PyAudio: https://people.csail.mit.edu/hubert/pyaudio/
- Pydub: https://github.com/jiaaro/pydub
- Noisereduce: https://github.com/timsainb/noisereduce

---

**Chúc bạn có trải nghiệm voice chat tuyệt vời! 🎤🤖**
