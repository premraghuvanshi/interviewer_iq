from gtts import gTTS
import os

print("Generating audio file...")
try:
    text = "Audio system is working perfectly."
    tts = gTTS(text=text, lang='en')
    
    # Save inside the tests folder
    output_path = os.path.join(os.path.dirname(__file__), "test_audio.mp3")
    tts.save(output_path)
    
    print(f"Success! Audio saved to: {output_path}")
except Exception as e:
    print("Error generating TTS:", e)