# 🎤 Voice Chat - Quick Start

## ⚡ Setup nhanh (5 phút)

### 1. Cài dependencies
```bash
pip install groq elevenlabs langdetect noisereduce pyaudio pydub
```

### 2. Thêm API keys vào `.env`
```env
GROQ_API_KEY=your_groq_key_here
ELEVEN_API_KEY=your_elevenlabs_key_here
```

### 3. Khởi động services
```bash
# Start DB và Qdrant
docker-compose up -d db qdrant

# Ingest data (nếu chưa)
python ingest_data.py

# Start app
uvicorn app.main:app --reload
```

### 4. Mở Voice Chat
```
http://localhost:8000/static/voice_chat.html
```

---

## 🎯 Cách dùng

1. **Click "+ New Chat"** để tạo session
2. **Click 🎤** hoặc nhấn **Space** để bắt đầu nói
3. **Nói câu hỏi** của bạn
4. **Click 🎤** lại hoặc nhấn **Space** để dừng
5. **Đợi AI trả lời**

---

## 🔑 Lấy API Keys

### Groq (STT - FREE)
1. Truy cập: https://console.groq.com/
2. Sign up
3. Tạo API key
4. Copy vào `.env`

### ElevenLabs (TTS - FREE tier)
1. Truy cập: https://elevenlabs.io/
2. Sign up
3. Vào Settings > API Keys
4. Copy vào `.env`

---

## 🎨 Features

- ✅ Voice-to-text (Groq Whisper)
- ✅ RAG search trong vector store
- ✅ Text-to-speech response
- ✅ Lưu vào database
- ✅ Đổi tên session
- ✅ Chuyển đổi Text/Voice mode
- ✅ Keyboard shortcuts

---

## 🐛 Troubleshooting

### Microphone không hoạt động
- Cho phép browser access microphone
- Reload page

### "GROQ_API_KEY not found"
```bash
# Kiểm tra .env
cat .env | grep GROQ

# Restart app
uvicorn app.main:app --reload
```

### "Vector store not initialized"
```bash
python ingest_data.py
```

---

## 📖 Full Guide

Xem chi tiết: `VOICE_CHAT_GUIDE.md`

---

**Ready to chat! 🎤**
