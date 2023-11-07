import pyttsx3
import speech_recognition
from date import datetime
import speech_recognition as sr
import pyttsx3 as tts
import webbrowser
import wikipedia
import wolframalpha


engine = pyttsx3.init()
voices = engine.setProperty('voices')
engine.setProperty('voice',voices[0].id) # 0 - male, 1 - female
activationWord = 'computer'


def speak(text, rate=150):
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()


def parsecommand():
    listener = sr.Recognizer()
    print("Listening for a command....")
    with sr.Microphone() as source:
        listener.pause_threshold = 2
        input_speech = listener.listen(source)


    try:
        print('Recongnizing speech...')
        query = listener.recognize_google()
        print(f"The input speech was: {query}")
    except Exception as exception:
        print("I didn't quite catch that")
        speak("I didn't quite catch that")
        print(exception)
        return None
    return query
