import os
import cv2
import numpy as np
import pyaudio
import wave
import threading
import time
import tempfile
import re
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play
from queue import Queue
from scipy.signal import butter, filtfilt

from src.processors.speaker import Speaker
from src.utils.audio_recorder import AudioRecorder
from src.processors.heavy_ocr import HeavyOCRProcessor
from src.processors.light_ocr import LightOCRProcessor
from src.processors.transcriber import AudioTranscriber



def save_frame_to_jpg(frame):
    """Save frame to jpg file and return the file path"""
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_img.name, frame)
    return temp_img.name


def find_best_text_frame(cap, light_ocr, duration=10):
    """Capture frames for duration seconds and find the one with most text"""
    start_time = time.time()
    best_frame = None
    best_text_count = 0
    best_text = ""
    frames_analyzed = 0
    
    print(f"Capturing frames for {duration} seconds to find text...")
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frames_analyzed += 1
        
        # Save frame to file for OCR processing
        frame_path = save_frame_to_jpg(frame)
        
        try:
            # Use light OCR to detect text
            detected_text = light_ocr.process(frame_path)
            os.unlink(frame_path)  # Clean up
            
            if detected_text:
                # Count words as a simple metric
                word_count = len(re.findall(r'\w+', detected_text))
                
                if word_count > best_text_count:
                    best_text_count = word_count
                    best_frame = frame.copy()
                    best_text = detected_text
                    print(f"Frame {frames_analyzed}: Found {word_count} words - New best frame!")
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            if os.path.exists(frame_path):
                os.unlink(frame_path)
        
        # Show frame with "Searching for text..." overlay
        info_frame = frame.copy()
        cv2.putText(info_frame, "Searching for text...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live_Feed', info_frame)
        cv2.waitKey(1)
    
    print(f"Best frame found with {best_text_count} words")
    return best_frame, best_text


def process_audio(audio_file, transcriber):
    """Process audio file and return transcription"""
    try:
        transcription = transcriber.process(audio_file)
        print(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"Error in transcription: {e}")
        return ""


def play_audio_file(audio_path):
    """Play audio file using pydub"""
    try:
        print(f"Playing audio: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        play(audio)
    except Exception as e:
        print(f"Error playing audio: {e}")


def process(light_ocr, heavy_ocr, transcriber, speaker):
    """Main processing function"""
    print("Initializing system...")
    
    # Initialize camera
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Check camera resolution
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {frame_width}x{frame_height}")
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return
    
    # Initialize audio recorder with error handling
    try:
        # Set up recorder with 10-second interval
        recorder = AudioRecorder(threshold=6000, recording_interval=10)
        recorder.start_listening()
    except Exception as e:
        print(f"Failed to initialize audio recorder: {e}")
        cap.release()
        return
    
    try:
        print("System initializing. Please wait for noise calibration...")
        # Wait for calibration to complete
        while not recorder.is_calibrated:
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, "Calibrating noise...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Live_Feed', frame)
                cv2.waitKey(1)
            time.sleep(0.1)
            
        print("System ready! Recording audio every 10 seconds.")
        print("Say 'What is written here' to trigger OCR processing.")
        print("Press 'q' to quit.")
        
        last_status_time = time.time()
        status_msg = ""
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Display status message
            current_time = time.time()
            if current_time - last_status_time > 5:  # Update status message every 5 seconds
                seconds_until_next = max(0, recorder.recording_interval - (current_time - recorder.last_recording_time))
                status_msg = f"Next recording in: {int(seconds_until_next)}s"
                last_status_time = current_time
            
            # Display info on frame
            info_frame = frame.copy()
            cv2.putText(info_frame, status_msg, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Live_Feed', info_frame)
            
            # Check if there's audio to process
            if not recorder.audio_queue.empty():
                audio_file = recorder.audio_queue.get()
                
                # Transcribe the audio
                transcription = process_audio(audio_file, transcriber)
                
                # Check if user is asking about text
                if transcription and re.search(r'what is written (here|there)', transcription.lower()):
                    print("OCR request detected!")
                    
                    # Find frame with most text
                    best_frame, light_text = find_best_text_frame(cap, light_ocr)
                    
                    if best_frame is not None:
                        # Save frame to temporary file for heavy OCR
                        temp_img = save_frame_to_jpg(best_frame)
                        
                        try:
                            # Process with heavy OCR
                            ocr_text = heavy_ocr.process(temp_img)
                            print(f"OCR Result: {ocr_text}")
                            
                            if ocr_text:
                                # Generate speech from OCR text
                                audio_path = speaker.process(ocr_text)
                                
                                # Play the audio
                                play_audio_file(audio_path)
                            else:
                                print("No text detected in the selected frame")
                        except Exception as e:
                            print(f"Error in heavy OCR processing: {e}")
                        
                        finally:
                            if os.path.exists(temp_img):
                                os.unlink(temp_img)  # Clean up
                
                # Clean up the audio file
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
            
            # Exit on 'q' key press
            if cv2.waitKey(30) == ord("q"):
                print("Quit command received")
                break
    
    except Exception as e:
        print(f"Error in main processing loop: {e}")
    
    finally:
        print("Cleaning up resources...")
        # Clean up
        try:
            recorder.stop()
        except Exception as e:
            print(f"Error stopping recorder: {e}")
        
        try:
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error releasing camera: {e}")
        
        print("System shutdown complete")


if __name__ == "__main__":
    try:
        print("Initializing OCR and audio processing components...")
        light_ocr = LightOCRProcessor(tesseract_cmd='/usr/bin/tesseract', lang='eng', config='--psm 6')
        heavy_ocr = HeavyOCRProcessor(use_angle=True, lang='en')
        transcriber = AudioTranscriber(language="en-US")
        speaker = Speaker()
        
        process(light_ocr, heavy_ocr, transcriber, speaker)
    except Exception as e:
        print(f"Fatal error in main program: {e}")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()