import streamlit as st
import os
import io
import time
from gtts import gTTS
from textblob import TextBlob
from dotenv import load_dotenv, find_dotenv
from src.ml.matcher import RoleMatcher
from src.agent.interviewer import InterviewAgent
import speech_recognition as sr

# Load environment variables
load_dotenv(find_dotenv())

# --- UI Configuration & Professional Styling ---
st.set_page_config(page_title="Interviewer IQ", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stTextArea textarea { border-radius: 8px; border: 1px solid #e0e0e0; }
    .stButton>button { 
        border-radius: 20px; 
        width: 100%; 
        border: 1px solid #1a1a1a; 
        background-color: white;
        color: #1a1a1a;
        transition: 0.2s; 
    }
    .stButton>button:hover { background-color: #1a1a1a; color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if "step" not in st.session_state:
    st.session_state.step = "setup"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Stores: role, content, sentiment, wpm
if "agent" not in st.session_state:
    st.session_state.agent = None
if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = None

audio_placeholder = st.empty()

# --- Core Helper Functions ---

def get_sentiment(text):
    """Analyzes the emotional tone using TextBlob."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1: return "Confident/Positive"
    elif polarity < -0.1: return "Hesitant/Negative"
    return "Neutral/Professional"

def speak(text):
    """Generates audio for AI responses."""
    if st.session_state.last_spoken != text:
        try:
            tts = gTTS(text=text, lang='en')
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            audio_placeholder.audio(audio_bytes, format='audio/mp3', autoplay=True)
            st.session_state.last_spoken = text
        except Exception:
            pass

def get_voice_input():
    """Captures and transcribes microphone input with hardware stability fixes."""
    r = sr.Recognizer()
    
    # Use a specific sample_rate and chunk_size to prevent buffer errors on 2nd use
    with sr.Microphone(sample_rate=48000, chunk_size=1024) as source:
        try:
            # Shorten calibration to prevent resource locking
            r.adjust_for_ambient_noise(source, duration=0.5)
            st.toast("🎤 Listening...")
            
            start_time = time.time()
            # Added a timeout so the mic doesn't hang if no sound is detected
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            end_time = time.time()
            
            text = r.recognize_google(audio)
            
            # Calculate WPM for behavioral analytics
            duration = end_time - start_time
            wpm = len(text.split()) / (duration / 60) if duration > 0 else 0
            
            return text, round(wpm)
        except Exception as e:
            # If the mic fails, the app won't crash; it will just show this error
            st.error(f"Mic error: {e}")
            return None, 0

# --- Application Flow ---

# 1. SETUP PHASE
if st.session_state.step == "setup":
    st.title("⚖️ Interviewer IQ")
    st.caption("Professional AI Technical Assessment Platform")
    st.markdown("---")
    
    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.subheader("Candidate Background")
        user_skills = st.text_area(
            "Paste your skills, experience, or project summaries:", 
            height=300,
            placeholder="e.g., AIML student at JIT. Experience with Python and Scikit-learn."
        )
    
    with right_col:
        st.markdown("### How it works")
        st.info("1. **Role Matching**: ML identification of target role.\n2. **Assessment**: Technical interview with behavioral tracking.\n3. **Feedback**: Tabbed report including Sentiment and Speaking Rate.")
        if st.button("🚀 Start Technical Interview"):
            if user_skills:
                with st.spinner("Analyzing profile..."):
                    matcher = RoleMatcher()
                    role = matcher.predict_role(user_skills) 
                    st.session_state.agent = InterviewAgent(role) 
                    st.session_state.matched_role = role
                    st.session_state.step = "interview"
                    st.rerun()
            else:
                st.warning("Please provide skills to proceed.")

# 2. INTERVIEW PHASE
elif st.session_state.step == "interview":
    with st.sidebar:
        st.header("Session Control")
        st.success(f"**Target Role:**\n{st.session_state.matched_role}")
        st.markdown("---")
        if st.button("🛑 End Interview & View Report"):
            st.session_state.step = "evaluation"
            st.rerun()

    st.subheader("Technical Screening Panel")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if not st.session_state.chat_history:
        with st.spinner("Preparing first question..."):
            first_q = st.session_state.agent.get_next_question()
            st.session_state.chat_history.append({"role": "assistant", "content": first_q, "sentiment": "N/A", "wpm": 0})
            st.rerun()

    if st.session_state.chat_history[-1]["role"] == "assistant":
        speak(st.session_state.chat_history[-1]["content"])

    st.markdown("---")
    chat_col, mic_col = st.columns([6, 1])
    
    user_text, current_wpm = None, 0

    with mic_col:
        if st.button("🎤 Mic"):
            user_text, current_wpm = get_voice_input()
            
    with chat_col:
        input_text = st.chat_input("Your answer...")
        if input_text: user_text = input_text

    if user_text:
        sentiment = get_sentiment(user_text)
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_text, 
            "sentiment": sentiment, 
            "wpm": current_wpm
        })
        with st.spinner("AI is thinking..."):
            ai_resp = st.session_state.agent.get_next_question(user_text)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_resp, "sentiment": "N/A", "wpm": 0})
        st.rerun()

# 3. EVALUATION PHASE
elif st.session_state.step == "evaluation":
    st.title("📊 Performance Analysis")
    st.markdown("---")
    
    with st.spinner("Finalizing Analytics..."):
        transcript_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
        report = st.session_state.agent.get_feedback(transcript_text) 
        
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Executive Summary", "🧠 Behavioral Insights", "📝 Transcript", "💡 Next Steps"])

    with tab1:
        st.subheader("Professional Feedback")
        st.markdown(report) 

    with tab2:
        st.subheader("Soft Skills & Sentiment Tracking")
        user_msgs = [m for m in st.session_state.chat_history if m["role"] == "user"]
        
        if user_msgs:
            # Calculate Average WPM from valid voice inputs
            voice_inputs = [m["wpm"] for m in user_msgs if m["wpm"] > 0]
            avg_wpm = sum(voice_inputs) / len(voice_inputs) if voice_inputs else 0
            
            c1, c2 = st.columns(2)
            c1.metric("Avg Speaking Rate", f"{round(avg_wpm)} WPM")
            c2.metric("Final Sentiment", user_msgs[-1]["sentiment"])

            st.write("**Response-by-Response Analysis:**")
            for i, m in enumerate(user_msgs):
                st.info(f"**Answer {i+1}:** {m['sentiment']} | Speed: {m['wpm']} WPM")
        else:
            st.info("No candidate responses found for analysis.")

    with tab3:
        st.subheader("Interview Dialogue")
        for i, message in enumerate(st.session_state.chat_history):
            role_label = "Interviewer" if message["role"] == "assistant" else "Candidate"
            st.text_area(
                label=f"{role_label} {i+1}", 
                value=message["content"], 
                height=100, 
                disabled=True,
                key=f"hist_{i}" # Unique key fix
            )

    with tab4:
        st.subheader("Preparation Roadmap")
        st.info(f"Matched Role: {st.session_state.matched_role}")
        if st.button("🔄 Start New Session"):
            st.session_state.step = "setup"
            st.session_state.chat_history = []
            st.session_state.last_spoken = None
            st.rerun()