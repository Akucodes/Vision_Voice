import os
import time
import wave
import threading
import tempfile
from queue import Queue
from datetime import datetime

import numpy as np
from scipy.signal import butter, filtfilt
import pyaudio



class AudioRecorder:
    def __init__(self, threshold=6000, chunk_size=1024, format=pyaudio.paInt16, 
                 channels=1, rate=44100, recording_interval=10):
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.format = format
        self.channels = channels
        self.rate = rate
        self.recording_interval = recording_interval  # Record every X seconds
        self.p = None
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.audio_queue = Queue()
        self.running = True
        
        # Parameters for noise filtering
        self.noise_floor = 0
        self.noise_calibration_samples = []
        self.calibration_period = 3  # seconds
        self.is_calibrated = False
        
        # Parameters for automatic recording
        self.last_recording_time = 0
        
        # Parameters for speech detection
        self.speech_frames_threshold = 3  # Minimum consecutive frames above threshold to detect speech
        self.consecutive_speech_frames = 0

    def start_listening(self):
        """Start listening for audio above threshold"""
        try:
            self.p = pyaudio.PyAudio()
            # Get default input device info
            device_info = self.p.get_default_input_device_info()
            print(f"Using audio device: {device_info['name']}")
            
            # Open stream with proper error handling
            self.stream = self.p.open(format=self.format,
                                     channels=self.channels,
                                     rate=self.rate,
                                     input=True,
                                     frames_per_buffer=self.chunk_size,
                                     start=False)  # Don't start yet
            
            # Start the stream explicitly after configuration
            self.stream.start_stream()
            
            # Start calibration
            calibration_thread = threading.Thread(target=self._calibrate_noise)
            calibration_thread.daemon = True
            calibration_thread.start()
            
            print("Audio recording system initialized successfully")
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            if self.p:
                self.p.terminate()
    
    def _calibrate_noise(self):
        """Calibrate noise floor by sampling ambient sound"""
        print(f"Calibrating noise floor for {self.calibration_period} seconds...")
        print("Please remain silent during calibration...")
        
        start_time = time.time()
        
        while time.time() - start_time < self.calibration_period:
            try:
                if self.stream and self.stream.is_active():
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume = np.max(np.abs(audio_data))
                    self.noise_calibration_samples.append(volume)
                time.sleep(0.01)
            except Exception as e:
                print(f"Error during calibration: {e}")
        
        # Calculate noise floor as mean + 2 standard deviations
        if self.noise_calibration_samples:
            mean_noise = np.mean(self.noise_calibration_samples)
            std_noise = np.std(self.noise_calibration_samples)
            self.noise_floor = mean_noise + 2 * std_noise
            
            # Adjust threshold based on noise floor
            self.threshold = max(self.threshold, self.noise_floor * 1.5)
            
            print(f"Calibration complete. Noise floor: {self.noise_floor:.2f}")
            print(f"Adjusted threshold: {self.threshold:.2f}")
        else:
            print("Calibration failed. Using default threshold.")
        
        self.is_calibrated = True
        
        # Start listening thread after calibration
        self.listening_thread = threading.Thread(target=self._listen_and_record)
        self.listening_thread.daemon = True
        self.listening_thread.start()
    
    def _butter_bandpass(self, lowcut=300, highcut=3000, fs=44100, order=5):
        """Design a bandpass filter to focus on human speech frequencies"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def _filter_audio(self, data):
        """Apply bandpass filter to audio data to focus on speech frequencies"""
        b, a = self._butter_bandpass()
        y = filtfilt(b, a, data)
        return y
        
    def _listen_and_record(self):
        """Listen for audio and record at fixed intervals"""
        print("Audio monitoring started...")
        
        # Wait for calibration to complete
        while not self.is_calibrated:
            time.sleep(0.1)
        
        # Initialize recording time
        self.last_recording_time = time.time()
        
        temp_frames = []  # Buffer for current interval's audio
        record_start_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for a new recording
                if current_time - self.last_recording_time >= self.recording_interval:
                    print(f"Recording interval reached: {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Only save if we have enough data
                    if len(temp_frames) > 0:
                        self._save_recording(temp_frames)
                    
                    # Reset for next interval
                    temp_frames = []
                    self.last_recording_time = current_time
                    record_start_time = current_time
                
                # Continuously read audio
                if self.stream and self.stream.is_active():
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Add raw data to buffer
                    temp_frames.append(data)
                    
                    # Process for speech detection
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    filtered_data = self._filter_audio(audio_data)
                    volume = np.max(np.abs(filtered_data))
                    
                    # Check for speech patterns (sustained volume above threshold)
                    if volume > self.threshold:
                        self.consecutive_speech_frames += 1
                        if self.consecutive_speech_frames == self.speech_frames_threshold:
                            print(f"Speech detected: Volume {volume:.2f} > Threshold {self.threshold:.2f}")
                    else:
                        self.consecutive_speech_frames = 0
                
                time.sleep(0.01)
                
            except IOError as e:
                print(f"Stream read error (non-fatal): {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in listening thread: {e}")
                time.sleep(0.5)
    
    def _save_recording(self, frames):
        """Save the recorded frames to a temporary file"""
        # Save the recorded audio to a temporary file
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            
            wf = wave.open(temp_file.name, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Add to queue for processing
            self.audio_queue.put(temp_file.name)
            print(f"Recording saved to {temp_file.name}")
            
        except Exception as e:
            print(f"Error saving audio file: {e}")
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        
    def stop(self):
        """Stop recording and close stream"""
        self.running = False
        time.sleep(0.2)  # Give threads time to notice
        
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")
        
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")