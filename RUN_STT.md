# 🎤 Hướng Dẫn Chạy Speech-to-Text

## 📋 Checklist Trước Khi Chạy

- [ ] Đã cài dependencies: `pip install -r requirements-voice.txt`
- [ ] Đã có GROQ_API_KEY trong `.env`
- [ ] Đã cài FFmpeg (nếu dùng MP3/M4A)
- [ ] Đã có file audio test (hoặc microphone)

## 🚀 Cách Chạy

### **Cách 1: Chạy từ thư mục gốc (KHUYẾN NGHỊ)**

```powershell
# Từ D:\downloads\work_8seneca\mini_pj
python -m app.service.speech.stt
```

**Giải thích**: `-m` chạy module, Python tự động thêm thư mục gốc vào path

### **Cách 2: Set PYTHONPATH**

```powershell
# PowerShell
$env:PYTHONPATH = "D:\downloads\work_8seneca\mini_pj"
python app/service/speech/stt.py
```

### **Cách 3: Chạy từ Python REPL**

```python
# Từ thư mục gốc
python

>>> from app.service.speech.stt import GroqSTT
>>> stt = GroqSTT()
>>> result = stt.transcribe_file("data/t1.wav", language="en")
>>> print(result['text'])
```

## 📁 Cấu Trúc File

```
mini_pj/
├── app/
│   ├── service/
│   │   └── speech/
│   │       ├── stt.py          # ← File này
│   │       └── tts.py
│   └── utils/
│       └── logger.py
├── data/
│   └── t1.wav                  # ← File audio test
├── .env                        # ← GROQ_API_KEY ở đây
└── requirements-voice.txt
```

## 🎯 Demo Options

Khi chạy, chọn:

### **1. Transcribe audio file**
- Transcribe file có sẵn
- Nhanh, không cần microphone
- Hỗ trợ: WAV, MP3, M4A, FLAC, OGG

### **2. Real-time recording**
- Thu âm từ microphone
- Tự động dừng sau 2s im lặng
- Cần microphone hoạt động

### **3. Both**
- Chạy cả 2 demo

## ⚙️ Configuration

### Thay đổi file audio

Trong `stt.py`, dòng:
```python
file_path = "data/t1.wav"
```

Đổi thành:
```python
file_path = "path/to/your/audio.wav"
```

### Thay đổi ngôn ngữ

```python
# Tiếng Việt
result = stt.transcribe_file(file_path, language="vi")

# Tiếng Anh
result = stt.transcribe_file(file_path, language="en")

# Auto-detect
result = stt.transcribe_file(file_path, language=None)
```

### Điều chỉnh VAD (Voice Activity Detection)

Trong `stt.py`, đầu file:
```python
THRESHOLD = 500          # Tăng nếu bắt nhiễu, giảm nếu không bắt giọng
SILENCE_LIMIT = 2.0      # Thời gian im lặng trước khi dừng (giây)
```

## 🐛 Troubleshooting

### Lỗi: "No module named 'app'"

**Nguyên nhân**: Chạy file từ thư mục con

**Giải pháp**:
```powershell
# Chạy từ thư mục gốc
cd D:\downloads\work_8seneca\mini_pj
python -m app.service.speech.stt
```

### Lỗi: "GROQ_API_KEY not found"

**Giải pháp**:
1. Kiểm tra file `.env` có dòng: `GROQ_API_KEY=gsk_...`
2. Restart terminal
3. Hoặc set trực tiếp:
   ```powershell
   $env:GROQ_API_KEY = "gsk_your_key_here"
   ```

### Lỗi: "Couldn't find ffmpeg"

**Giải pháp**:
1. Cài FFmpeg (xem `FFMPEG_SETUP.md`)
2. Hoặc chỉ dùng file WAV (không cần FFmpeg)

### Lỗi: "File not found: data/t1.wav"

**Giải pháp**:
1. Tạo thư mục `data/`
2. Đặt file audio vào đó
3. Hoặc đổi đường dẫn trong code

### Lỗi: "No default input device"

**Nguyên nhân**: Không có microphone hoặc chưa cấp quyền

**Giải pháp**:
1. Kết nối microphone
2. Cấp quyền microphone cho terminal
3. Chọn demo 1 (file) thay vì demo 2 (recording)

### Warning: "RuntimeWarning: Couldn't find ffmpeg"

**Không ảnh hưởng** nếu bạn chỉ dùng file WAV. Nếu muốn dùng MP3/M4A, cài FFmpeg.

## 📊 Expected Output

```
🎤 Speech-to-Text Demo (Groq + Whisper large-v3)

Choose demo:
1. Transcribe audio file
2. Real-time recording
3. Both

Enter choice (1/2/3): 1

2024-12-04 10:30:15 - stt - ℹ️  INFO - Groq STT initialized whisper
2024-12-04 10:30:15 - stt - ℹ️  INFO - Preprocessing audio...
2024-12-04 10:30:16 - stt - ℹ️  INFO - Done in 0.85s
2024-12-04 10:30:16 - stt - ℹ️  INFO - Text: Hello, this is a test.
2024-12-04 10:30:16 - stt - ℹ️  INFO - Duration: 3.50s
2024-12-04 10:30:16 - stt - ℹ️  INFO - Processing: 0.85s
2024-12-04 10:30:16 - stt - ℹ️  INFO - Speed: 4.1x real-time
```

## 🎨 Log Colors

- 🔍 **DEBUG** - Gray
- ℹ️  **INFO** - Blue
- ⚠️  **WARNING** - Yellow
- ❌ **ERROR** - Red
- 🔥 **CRITICAL** - White on Red

## 📝 Next Steps

1. ✅ Test với file audio
2. ✅ Test với microphone
3. ✅ Tích hợp vào FastAPI
4. ✅ Thêm caching
5. ✅ Deploy

## 🔗 Resources

- Groq Console: https://console.groq.com
- FFmpeg Download: https://www.gyan.dev/ffmpeg/builds/
- PyAudio Docs: https://people.csail.mit.edu/hubert/pyaudio/
