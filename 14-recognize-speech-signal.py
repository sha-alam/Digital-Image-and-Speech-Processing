import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Read the speech signal (replace 'speech.wav' with your audio file)
speech_signal = sr.AudioFile('sound.wav')

# Recognize speech using the Google Web Speech API
with speech_signal as source:
    audio_data = recognizer.record(source)  # Record the audio from the file

try:
    recognized_text = recognizer.recognize_google(audio_data)
    print("Recognized Text:", recognized_text)
except sr.UnknownValueError:
    print("Speech recognition could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Web Speech API; {e}")
