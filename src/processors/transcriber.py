import os
import speech_recognition as sr
from pydub import AudioSegment

class AudioTranscriber:
    def __init__(self, chunk_length_ms=60000, language="en-US"):
        self.chunk_length_ms = chunk_length_ms
        self.language = language
        self.recognizer = None
        self.load_model()

    def load_model(self):
        """
        Initialize and load any models or resources needed.
        """
        self.recognizer = sr.Recognizer()

    def process(self, file_path):
        """
        Process an audio file: load it, split into chunks if necessary, transcribe and return text.
        
        Args:
            file_path (str): Path to the audio file (.mp3, .wav, etc.)
        
        Returns:
            str: Transcribed text
        """
        if self.recognizer is None:
            raise Exception("Model not loaded. Call load_model() first.")

        # Load the audio file
        ext = file_path.split('.')[-1].lower()
        if ext == "mp3":
            audio = AudioSegment.from_mp3(file_path)
        elif ext in ("aiff", "aif"):
            audio = AudioSegment.from_file(file_path, format="aiff")
        elif ext == "wav":
            audio = AudioSegment.from_wav(file_path)
        elif ext == "avi":
            audio = AudioSegment.from_file(file_path, format="avi")
        else:
            raise ValueError("Unsupported file format. Please use .mp3, .wav, .aiff, or .avi files.")
        
        audio_chunks = [audio[i:i + self.chunk_length_ms] for i in range(0, len(audio), self.chunk_length_ms)]
        full_text = ""

        for i, chunk in enumerate(audio_chunks):
            chunk_file = f"chunk_{i}.wav"
            chunk.export(chunk_file, format="wav")
            
            with sr.AudioFile(chunk_file) as source:
                audio_data = self.recognizer.record(source)
                
                try:
                    text = self.recognizer.recognize_google(audio_data, language=self.language)
                    full_text += text + " "
                except sr.UnknownValueError:
                    print(f"Could not understand audio in chunk {i}.")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}.")
            
            os.remove(chunk_file)
        
        return full_text.strip()

# if __name__ == "__main__":
#     # Example usage
#     transcriber = AudioTranscriber(language="en-US")

#     file_path = '/home/akanksha/Downloads/output.mp3'
#     print(f"Processing {file_path}...")
#     text = transcriber.process(file_path)
#     print(f"Transcribed Text for {file_path}:\n{text}")
#     print("-" * 50)
