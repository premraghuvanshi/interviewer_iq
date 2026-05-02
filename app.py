import streamlit as st
import os
import io
import time
import json
import pandas as pd
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
    /* Main Background Contrast */
    .main { 
        background-color: #f8fafc; 
        font-family: 'Inter', sans-serif;
    }
    
    /* SIDEBAR VISIBILITY FIX */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e2e8f0;
    }

    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1e293b !important;
    }

    /* Metric Cards Styling */
    .metric-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border-top: 5px solid #6366f1;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-container h5 { color: #64748b !important; font-size: 0.9rem; text-transform: uppercase; }
    .metric-container h2 { color: #0f172a !important; font-weight: 800; }

    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #6366f1;
    }

    /* Primary Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if "step" not in st.session_state: st.session_state.step = "setup"
if "chat_history" not in st.session_state: st.session_state.chat_history = [] 
if "agent" not in st.session_state: st.session_state.agent = None
if "last_spoken" not in st.session_state: st.session_state.last_spoken = None
if "q_count" not in st.session_state: st.session_state.q_count = 1  # Question Counter

audio_placeholder = st.empty()

# --- Helper Functions ---

def safe_parse_json(ai_response):
    if isinstance(ai_response, dict): return ai_response
    try:
        cleaned = ai_response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception:
        return {
            "question": "Could you please clarify your last technical point?",
            "technical_accuracy": 0, "depth": 0, "communication": 0, "confidence": 0,
            "strengths": "N/A", "improvements": "N/A", "feedback": "Sync Error", "suggestion": "N/A"
        }

def speak(text):
    if st.session_state.last_spoken != text:
        try:
            tts = gTTS(text=text, lang='en')
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            audio_placeholder.audio(audio_bytes, format='audio/mp3', autoplay=True)
            st.session_state.last_spoken = text
        except: pass

def get_voice_input():
    r = sr.Recognizer()
    r.pause_threshold = 1.0 
    r.energy_threshold = 400 
    with sr.Microphone(sample_rate=48000, chunk_size=1024) as source:
        try:
            r.adjust_for_ambient_noise(source, duration=0.3)
            st.toast("🎤 Listening...", icon="🎙️")
            audio = r.listen(source, timeout=5, phrase_time_limit=20)
            text = r.recognize_google(audio)
            return text
        except Exception: return None

# --- Application Flow ---

# 1. SETUP
if st.session_state.step == "setup":
    st.title("⚖️ Interviewer IQ")
    st.subheader("Elite AI Technical Assessment")
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        user_skills = st.text_area("Candidate Profile & Skills", height=300, 
                                 placeholder="e.g., 3rd year AIML student at JIT. Skills: Python, ML, NLP...")
        if st.button("🚀 LAUNCH INTERVIEW"):
            if user_skills:
                with st.spinner("Analyzing Background..."):
                    matcher = RoleMatcher()
                    role = matcher.predict_role(user_skills)
                    st.session_state.agent = InterviewAgent(role)
                    st.session_state.matched_role = role
                    st.session_state.step = "interview"
                    st.rerun()
            else:
                st.warning("Please provide skills to proceed.")
    with col_r:
        st.info("The system will evaluate you over 20 questions with increasing difficulty.")

# 2. INTERVIEW
elif st.session_state.step == "interview":
    with st.sidebar:
        st.title("📊 Progress")
        st.write(f"Question **{st.session_state.q_count}** of 20")
        st.progress(st.session_state.q_count / 20)
        
        st.markdown("---")
        st.success(f"**Target Role:**\n{st.session_state.matched_role}")
        
        # Live Stats from JSON
        ai_data_points = [m["analytics"] for m in st.session_state.chat_history if m["role"] == "assistant" and "analytics" in m]
        if ai_data_points:
            latest = ai_data_points[-1]
            st.metric("Tech Accuracy", f"{latest.get('technical_accuracy', 0)}/10")
            st.metric("Knowledge Depth", f"{latest.get('depth', 0)}/10")
            
            acc_scores = [d.get("technical_accuracy", 0) for d in ai_data_points]
            st.line_chart(pd.DataFrame(acc_scores, columns=["Accuracy Trend"]))

        st.markdown("---")
        if st.button("🛑 STOP & ANALYZE"):
            st.session_state.step = "evaluation"
            st.rerun()

    # Chat Feed
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Initial Question
    if not st.session_state.chat_history:
        raw_data = st.session_state.agent.get_next_question(question_count=st.session_state.q_count)
        ai_data = safe_parse_json(raw_data)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_data['question'], "analytics": ai_data})
        st.rerun()

    speak(st.session_state.chat_history[-1]["content"])

    st.markdown("---")
    c_in, c_mic = st.columns([6, 1])
    user_text = None
    with c_mic:
        if st.button("🎤"): user_text = get_voice_input()
    with c_in:
        chat_in = st.chat_input("Explain your answer...")
        if chat_in: user_text = chat_in

    if user_text:
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        st.session_state.q_count += 1
        
        # Auto-end at 20 questions
        if st.session_state.q_count > 20:
            st.toast("Final Question Reached. Generating Report...")
            time.sleep(2)
            st.session_state.step = "evaluation"
            st.rerun()
        else:
            with st.spinner(f"Evaluating Response {st.session_state.q_count-1}..."):
                raw_data = st.session_state.agent.get_next_question(user_text, st.session_state.q_count)
                ai_data = safe_parse_json(raw_data)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_data['question'], "analytics": ai_data})
            st.rerun()

# 3. EVALUATION
elif st.session_state.step == "evaluation":
    st.title("📊 Technical Performance Audit")
    
    ai_msgs = [m for m in st.session_state.chat_history if m["role"] == "assistant"]
    final_stats = ai_msgs[-1]["analytics"]
    
    h1, h2, h3 = st.columns(3)
    with h1: st.markdown(f'<div class="metric-container"><h5>Tech Accuracy</h5><h2>{final_stats.get("technical_accuracy")}/10</h2></div>', unsafe_allow_html=True)
    with h2: st.markdown(f'<div class="metric-container"><h5>Knowledge Depth</h5><h2>{final_stats.get("depth")}/10</h2></div>', unsafe_allow_html=True)
    with h3: st.markdown(f'<div class="metric-container"><h5>Communication</h5><h2>{final_stats.get("communication")}/10</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    tabs = st.tabs(["📋 Executive Summary", "🧠 Behavioral Analysis", "🗺️ Roadmap"])
    
    with tabs[0]:
        transcript = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
        st.markdown(st.session_state.agent.get_feedback(transcript))
    
    with tabs[1]:
        st.subheader("Key Findings")
        st.write(f"**Strengths:** {final_stats.get('strengths')}")
        st.write(f"**Gaps:** {final_stats.get('improvements')}")
        st.info(f"**Interviewer Feedback:** {final_stats.get('feedback')}")
        
    with tabs[2]:
        st.success("### Suggested Learning Path")
        st.write(final_stats.get('suggestion'))
        if st.button("🔄 NEW SESSION"):
            st.session_state.step = "setup"
            st.session_state.chat_history = []
            st.session_state.q_count = 1
            st.rerun()