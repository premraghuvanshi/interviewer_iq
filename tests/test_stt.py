import speech_recognition as sr

r = sr.Recognizer()

try:
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Listening! Say a few words...")
        
        # Listen for up to 5 seconds
        audio = r.listen(source, timeout=5, phrase_time_limit=5)
        
        print("Processing speech...")
        text = r.recognize_google(audio)
        print("\nYou said:", text)

except sr.WaitTimeoutError:
    print("Error: Listening timed out while waiting for phrase to start")
except sr.UnknownValueError:
    print("Error: Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print(f"Error: Could not request results from Google service; {e}")
except Exception as e:
    print("Microphone error:", e)
    print("Hint: You may need to run 'pip install pyaudio' in your terminal.")