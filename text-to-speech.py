import speech_recognition as sr
import pyttsx3
import os
import subprocess

# Initialize the recognizer and engine
listener = sr.Recognizer()
try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    engine = None

def speak(text):
    if engine:
        engine.say(text)
        engine.runAndWait()
    else:
        print(f"Assistant: {text}")

def take_command():
    try:
        with sr.Microphone() as source:
            print("Listening...")
            # Adjust regarding ambient noise
            listener.adjust_for_ambient_noise(source)
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            print(f"User said: {command}")
            return command
    except sr.UnknownValueError:
        # print("Could not understand audio")
        pass
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service")
    except Exception as e:
        print(f"Error: {e}")
    return ""

import webbrowser

def open_app(app_name):
    # macOS open command
    try:
        # Browser handling
        if "safari" in app_name:
            subprocess.run(["open", "-a", "Safari"])
            speak("Opening Safari")
            return
        elif "chrome" in app_name:
            subprocess.run(["open", "-a", "Google Chrome"])
            speak("Opening Google Chrome")
            return

        subprocess.run(["open", "-a", app_name])
        speak(f"Opening {app_name}")
    except Exception as e:
        speak(f"Could not open {app_name}")

def run_assistant():
    print("Voice Assistant Started. Say 'stop' to exit.")
    speak("Voice Assistant Started")
    
    while True:
        command = take_command()
        
        if not command:
            continue

        if 'play' in command:
            song = command.replace('play', '').strip()
            speak('playing ' + song)
            # Open YouTube with search query
            webbrowser.open(f"https://www.youtube.com/results?search_query={song}")
            
        elif 'open' in command:
            target = command.replace('open', '').strip()
            
            # Website shortcuts
            if 'youtube' in target:
                speak("Opening YouTube")
                webbrowser.open("https://www.youtube.com")
            elif 'google' in target:
                speak("Opening Google")
                webbrowser.open("https://www.google.com")
            else:
                open_app(target)

        elif 'search' in command:
            query = command.replace('search', '').strip()
            speak(f"Searching Google for {query}")
            webbrowser.open(f"https://www.google.com/search?q={query}")
            
        elif 'stop' in command or 'exit' in command:
            speak("Goodbye")
            break

if __name__ == "__main__":
    run_assistant()
