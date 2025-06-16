# 🎙️ VisionVoice

**VisionVoice** is an intelligent real-time audio-visual processing system that combines computer vision, speech recognition, and text-to-speech capabilities. The system continuously monitors audio input and can perform OCR (Optical Character Recognition) on camera feed when triggered by voice commands, then reads the detected text aloud.

## 📦 Features

- 🎤 **Continuous Audio Monitoring** with voice activity detection (VAD)
- 🔊 **Real-time Speech Recognition** from live audio streams  
- 📷 **Dual OCR Processing** with both lightweight and heavy OCR engines
- 🎛️ **Intelligent Frame Selection** to find frames with maximum text content
- 🔊 **Text-to-Speech Output** for detected text
- 🎬 **Live Camera Feed** with visual status indicators
- 🧠 **Voice-Triggered Processing** with natural language commands

## 🚀 Installation Guide

### 🧱 System Dependencies

Run the following commands to prepare your system:

```bash
sudo apt-get update

# Install CUDA Deep Neural Network library (cuDNN)
sudo apt-get install libcudnn8 libcudnn8-dev libcudnn8-samples -y

# (Optional) Run cuDNN sample test
cp -r /usr/src/cudnn_samples_v8 $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN

# Install PortAudio (required for PyAudio)
sudo apt-get install portaudio19-dev

# Install mpg321 for audio playback (optional)
sudo apt-get install mpg321

# Install FFmpeg
sudo apt install ffmpeg -y

# Install Tesseract OCR
sudo apt install tesseract-ocr libtesseract-dev -y
```

### 🐍 Python Environment Setup (with Conda)

We recommend using a Conda environment for dependency isolation.

```bash
# Create and activate environment
conda create -n vision_voice python=3.10 -y
conda activate vision_voice
```

### 📦 Python Package Installation

Install required Python packages:

```bash
pip install -r requirements.txt
```

If needed, manually install additional OCR dependencies:

```bash
# Tesseract via conda (optional)
conda install -c conda-forge tesseract
```

**Ensure GPU support is available (optional but recommended):**

```bash
nvidia-smi
nvcc --version
```

## 📂 Project Structure

```
.
├── main.py                          # Entry point with main processing loop
├── requirements.txt                 # Python requirements
└── src/
    ├── processors/
    │   ├── speaker.py              # Text-to-speech processing
    │   ├── transcriber.py          # Audio transcription logic
    │   ├── heavy_ocr.py           # Advanced OCR processing
    │   └── light_ocr.py           # Lightweight OCR processing
    └── utils/
        └── audio_recorder.py       # Audio recording with VAD
```

## 🔄 System Workflow

The VisionVoice system follows a sophisticated multi-step workflow:

### 1. **System Initialization**
- **Camera Setup**: Initializes the default camera (index 0) and checks resolution
- **Audio Recorder**: Sets up continuous audio monitoring with a 10-second recording interval
- **Component Loading**: Initializes all processors (Light OCR, Heavy OCR, Transcriber, Speaker)

### 2. **Noise Calibration Phase**
```python
# Wait for calibration to complete
while not recorder.is_calibrated:
    # Display calibration status on camera feed
    cv2.putText(frame, "Calibrating noise...", (50, 50), ...)
```
- The system automatically calibrates background noise levels
- Visual feedback shows calibration progress on the camera feed
- This ensures accurate voice activity detection

### 3. **Continuous Monitoring Loop**
The main loop performs several concurrent operations:

#### **Audio Processing Pipeline**
- **Continuous Recording**: Audio is recorded in 10-second intervals
- **Voice Activity Detection**: Only processes audio when speech is detected
- **Speech Recognition**: Converts detected speech to text using the transcriber
- **Command Recognition**: Looks for trigger phrases like "What is written here"

#### **Visual Status Display**
```python
# Display countdown and status
seconds_until_next = max(0, recorder.recording_interval - (current_time - recorder.last_recording_time))
status_msg = f"Next recording in: {int(seconds_until_next)}s"
cv2.putText(info_frame, status_msg, (10, 30), ...)
```

### 4. **OCR Processing Workflow**
When the trigger phrase is detected, the system executes a sophisticated OCR pipeline:

#### **Frame Analysis Phase**
```python
def find_best_text_frame(cap, light_ocr, duration=10):
    # Capture frames for 10 seconds
    # Analyze each frame with lightweight OCR
    # Select frame with maximum text content
    # Return best frame and detected text
```

**Process:**
1. **Frame Capture**: Records frames for 10 seconds
2. **Lightweight Analysis**: Uses fast OCR to count words in each frame
3. **Best Frame Selection**: Chooses frame with highest word count
4. **Visual Feedback**: Shows "Searching for text..." overlay during analysis

#### **Heavy OCR Processing**
```python
# Process with heavy OCR for accuracy
ocr_text = heavy_ocr.process(temp_img)
```
- Uses advanced OCR engine for maximum accuracy
- Processes the selected best frame
- Handles complex text layouts and formatting

#### **Text-to-Speech Output**
```python
# Generate and play audio
audio_path = speaker.process(ocr_text)
play_audio_file(audio_path)
```
- Converts detected text to natural speech
- Plays audio output to user
- Handles audio file cleanup automatically

### 5. **Resource Management**
The system includes comprehensive cleanup procedures:

```python
# Automatic cleanup of temporary files
if os.path.exists(frame_path):
    os.unlink(frame_path)
    
# Proper resource disposal
recorder.stop()
cap.release()
cv2.destroyAllWindows()
```

## 🎯 Usage Instructions

### Starting the System
```bash
conda activate vision_voice
python3 main.py
```

### Using Voice Commands
1. **Wait for Calibration**: Allow the system to calibrate background noise
2. **Trigger OCR**: Say "What is written here" or "What is written there"
3. **Wait for Processing**: The system will analyze frames for 10 seconds
4. **Listen**: The detected text will be read aloud automatically
5. **Exit**: Press 'q' to quit the application

### Visual Indicators
- **Red Text**: "Calibrating noise..." - System is still calibrating
- **Green Text**: "Next recording in: Xs" - Normal operation with countdown
- **Green Text**: "Searching for text..." - OCR processing active

## ⚙️ Configuration Options

### Audio Settings
- **Recording Interval**: 10 seconds (configurable in AudioRecorder)
- **Threshold**: 6000 (voice activity detection sensitivity)
- **Language**: "en-US" (speech recognition language)

### OCR Settings
- **Light OCR**: Tesseract with PSM 6 mode for fast processing
- **Heavy OCR**: PaddleOCR with angle detection for accuracy
- **Analysis Duration**: 10 seconds for frame selection

## 🔧 Troubleshooting

### Common Issues

**Camera Not Found**
```bash
# Check available cameras
ls /dev/video*
# Or use v4l2-ctl
v4l2-ctl --list-devices
```

**Audio Issues**
```bash
# Check audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)}') for i in range(p.get_device_count())]"
```

**OCR Not Working**
```bash
# Verify Tesseract installation
tesseract --version
tesseract --list-langs
```

### Performance Tips
- Ensure good lighting for better OCR accuracy
- Keep text clearly visible in camera frame
- Minimize background noise during voice commands
- Use GPU acceleration when available


## 🧠 Technical Notes

- **Dual OCR Strategy**: Light OCR for frame selection, Heavy OCR for final processing
- **Intelligent Frame Selection**: Analyzes multiple frames to find optimal text content
- **Voice Activity Detection**: Reduces false triggers from background noise
- **Memory Management**: Automatic cleanup of temporary files and resources
- **Multi-threading**: Concurrent audio processing and video display
- **Error Handling**: Comprehensive exception handling throughout the pipeline
