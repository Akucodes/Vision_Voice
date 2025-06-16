from gtts import gTTS
import os

class Speaker:

    def __init__(self):

        pass
    

    def process(self, text, output_file = "output.mp3"):

        # Convert text to speech
        tts = gTTS(text=text, lang="en", slow=False)
        
        tts.save(output_file)

        return output_file