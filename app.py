import streamlit as st
import os
import io
from gtts import gTTS
from dotenv import load_dotenv, find_dotenv
from src.ml.matcher import RoleMatcher
from src.agent.interviewer import InterviewAgent
import speech_recognition as sr

# Load environment variables
load_dotenv(find_dotenv())

# --- App Configuration ---
st.set_page_config(page_title="AI Technical Interviewer", page_icon="🤖")
st.title("🤖 AI Technical Interviewer")

# Initialize session state variables
if "step" not in st.session_state:
    st.session_state.step = "setup"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = None

# Placeholder for the persistent audio player
audio_placeholder = st.empty()

def speak(text):
    """Generates and plays audio without creating multiple players."""
    # Only speak if it's a new message to avoid loops
    if st.session_state.last_spoken != text:
        try:
            tts = gTTS(text=text, lang='en')
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            audio_placeholder.audio(audio_bytes, format='audio/mp3', autoplay=True)
            st.session_state.last_spoken = text
        except Exception as e:
            st.error(f"TTS Error: {e}")

def get_voice_input():
    """Captures audio with improved accuracy settings."""
    r = sr.Recognizer()
    
    # IMPROVEMENT 1: Adjust sensitivity
    # Higher values make it less sensitive (useful in noisy rooms)
    r.energy_threshold = 300 
    
    # IMPROVEMENT 2: Give yourself more time to pause between words
    r.pause_threshold = 1.2 

    with sr.Microphone() as source:
        try:
            # IMPROVEMENT 3: Longer calibration for ambient noise
            with st.spinner("Calibrating for background noise..."):
                r.adjust_for_ambient_noise(source, duration=1)
            
            st.toast("🎤 Listening! You can speak now...")
            
            # IMPROVEMENT 4: Remove strict timeouts for longer answers
            audio = r.listen(source, timeout=None)
            
            with st.spinner("Transcribing..."):
                text = r.recognize_google(audio)
                return text
        except sr.UnknownValueError:
            st.warning("I couldn't hear any clear words. Please try again or type.")
            return None
        except Exception as e:
            st.error(f"Microphone Error: {e}")
            return None

# --- UI LOGIC ---

if st.session_state.step == "setup":
    st.subheader("Identify Your Target Role")
    st.write("I will analyze your skills to tailor the interview questions.")
    
    # Example input placeholder reflects user's actual background
    user_skills = st.text_area(
        "Paste your skills or bio:", 
        placeholder="e.g., I am a 3rd-year AIML student working on Python, Machine Learning, and NLP projects like Edu2Job."
    )
    
    if st.button("Start Interview"):
        if user_skills:
            with st.spinner("Analyzing skills..."):
                matcher = RoleMatcher()
                role = matcher.predict_role(user_skills)
                st.session_state.agent = InterviewAgent(role)
                st.session_state.matched_role = role
                st.session_state.step = "interview"
                st.rerun()
        else:
            st.warning("Please enter some skills first.")

elif st.session_state.step == "interview":
    st.sidebar.success(f"Role: {st.session_state.matched_role}")
    if st.sidebar.button("Restart Interview"):
        st.session_state.step = "setup"
        st.session_state.chat_history = []
        st.session_state.last_spoken = None
        st.rerun()

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initial Question
    if not st.session_state.chat_history:
        with st.spinner("Generating first question..."):
            first_q = st.session_state.agent.get_next_question()
            st.session_state.chat_history.append({"role": "assistant", "content": first_q})
            st.rerun()

    # Audio Playback for the latest Assistant message
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
        speak(st.session_state.chat_history[-1]["content"])

    # User Input Area
    st.divider()
    col1, col2 = st.columns([1, 5])
    
    with col1:
        if st.button("🎤 Speak"):
            text = get_voice_input()
            if text:
                st.session_state.chat_history.append({"role": "user", "content": text})
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.get_next_question(text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
    
    with col2:
        user_text = st.chat_input("Type your answer here...")
        if user_text:
            st.session_state.chat_history.append({"role": "user", "content": user_text})
            with st.spinner("Thinking..."):
                response = st.session_state.agent.get_next_question(user_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()