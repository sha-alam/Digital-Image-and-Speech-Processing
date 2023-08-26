from gtts import gTTS
import os
def text_to_speech ( text,output_filename='output.wav', lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_filename)
    os.system(f'start {output_filename}')  # Open the generated audio file

input_text = input("Enter the text you want to convert to speech: ")
text_to_speech(input_text)