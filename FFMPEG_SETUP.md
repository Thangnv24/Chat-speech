# 🎵 FFmpeg Setup Guide

## Vấn Đề

Lỗi: `Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work`

Pydub cần FFmpeg để xử lý audio files (mp3, m4a, etc.). Nếu chỉ dùng WAV thì không cần FFmpeg.

## Giải Pháp

### **Windows (KHUYẾN NGHỊ - Dễ nhất)**

#### Cách 1: Dùng Chocolatey (Nhanh nhất)
```powershell
# Cài Chocolatey nếu chưa có
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Cài FFmpeg
choco install ffmpeg
```

#### Cách 2: Download Manual
1. Download FFmpeg: https://www.gyan.dev/ffmpeg/builds/
   - Chọn: `ffmpeg-release-essentials.zip`
2. Giải nén vào: `C:\ffmpeg`
3. Thêm vào PATH:
   ```powershell
   # PowerShell (Admin)
   $env:Path += ";C:\ffmpeg\bin"
   [Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)
   ```
4. Restart terminal

#### Cách 3: Dùng Scoop
```powershell
scoop install ffmpeg
```

### **Linux**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg

# Arch
sudo pacman -S ffmpeg
```

### **macOS**

```bash
brew install ffmpeg
```

## Kiểm Tra

```bash
ffmpeg -version
```

Nếu thấy version info → Thành công!

## Alternative: Không Dùng FFmpeg

Nếu không muốn cài FFmpeg, chỉ dùng file WAV:

```python
# Trong stt.py, comment dòng này:
# audio = AudioSegment.from_file(file_path)

# Thay bằng:
audio = AudioSegment.from_wav(file_path)
```

Hoặc convert audio sang WAV trước khi xử lý bằng tool online.

## Troubleshooting

### Lỗi: "ffmpeg not found" sau khi cài

1. Restart terminal
2. Kiểm tra PATH:
   ```powershell
   $env:Path -split ';' | Select-String ffmpeg
   ```
3. Nếu không có, thêm lại vào PATH

### Lỗi: "Permission denied"

Chạy PowerShell/CMD as Administrator

### Lỗi: "Cannot find the path"

Kiểm tra đường dẫn FFmpeg đúng chưa:
```powershell
Get-Command ffmpeg
```
