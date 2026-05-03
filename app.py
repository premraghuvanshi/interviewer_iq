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
from src.agent.interviewer import InterviewAgent, TutorAgent 
import speech_recognition as sr

# Load environment variables
load_dotenv(find_dotenv())

# --- UI Configuration & Tech-Vibrant Styling ---
st.set_page_config(page_title="InterviewIQ", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    .main { 
        background-color: #f1f5f9; 
        font-family: 'Inter', sans-serif;
    }
    
    .step-tracker { 
        display: flex; 
        justify-content: space-around; 
        padding: 15px; 
        background: white; 
        border-radius: 12px; 
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    .step { font-weight: 700; color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; }
    .step-active { color: #4f46e5; border-bottom: 3px solid #4f46e5; }

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #cbd5e1;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }

    .metric-container {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        border-top: 6px solid #4f46e5;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 10px 0;
        border: 1px solid #e2e8f0;
    }
    
    .metric-container h5 { color: #64748b !important; font-size: 0.9rem; font-weight: 600; margin-bottom: 10px;}
    .metric-container h2 { color: #1e293b !important; font-weight: 800; font-size: 2.8rem; margin: 0;}

    .stProgress > div > div > div > div { background-color: #4f46e5; }

    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px rgba(79, 70, 229, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if "step" not in st.session_state: st.session_state.step = "setup"
if "chat_history" not in st.session_state: st.session_state.chat_history = [] 
if "agent" not in st.session_state: st.session_state.agent = None
if "last_spoken" not in st.session_state: st.session_state.last_spoken = None
if "q_count" not in st.session_state: st.session_state.q_count = 1 
if "total_marks" not in st.session_state: st.session_state.total_marks = 0.0

audio_placeholder = st.empty()

# --- Helpers ---

def safe_parse_json(ai_response):
    if isinstance(ai_response, dict): return ai_response
    try:
        content = ai_response.strip()
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx + 1]
        return json.loads(content)
    except Exception as e:
        return {
            "question": "System sync error. Could you repeat your last technical point?",
            "score": 0.0, "technical_accuracy": 0, "depth": 0, "communication": 0, "confidence": 0,
            "strengths": "N/A", "improvements": str(e), "feedback": "JSON Error", "suggestion": "N/A"
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
    r.pause_threshold = 2.0  # Core timeout feature
    r.non_speaking_duration = 1.0 

    with sr.Microphone(sample_rate=48000) as source:
        try:
            r.adjust_for_ambient_noise(source, duration=0.3)
            st.toast("🎤 Listening... (Stops after 2s silence)", icon="🎙️")
            audio = r.listen(source, timeout=10, phrase_time_limit=45)
            return r.recognize_google(audio)
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
            return None
        except Exception as e:
            st.error(f"Mic Error: {str(e)}")
            return None

# --- Application Flow ---

# 1. SETUP
if st.session_state.step == "setup":
    st.markdown('<div class="step-tracker"><span class="step step-active">1. PROFILE</span><span class="step">2. INTERVIEW</span><span class="step">3. AUDIT</span></div>', unsafe_allow_html=True)
    st.title("⚖️ InterviewIQ")
    
    col_l, col_r = st.columns([3, 2], gap="large")
    with col_l:
        user_skills = st.text_area("Technical Background", height=300, placeholder="e.g., AIML Student at JIT, Python, SQL...")
        if st.button("🚀 START ASSESSMENT"):
            if user_skills:
                matcher = RoleMatcher()
                role = matcher.predict_role(user_skills)
                st.session_state.agent = InterviewAgent(role)
                st.session_state.matched_role = role
                st.session_state.total_marks = 0.0 # Reset for new session
                st.session_state.step = "interview"
                st.rerun()

# 2. INTERVIEW
elif st.session_state.step == "interview":
    st.markdown('<div class="step-tracker"><span class="step">1. PROFILE</span><span class="step step-active">2. INTERVIEW</span><span class="step">3. AUDIT</span></div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("📋 Live Progress")
        st.write(f"Question **{st.session_state.q_count}** / 20")
        st.progress(st.session_state.q_count / 20)
        
        ai_data_points = [m["analytics"] for m in st.session_state.chat_history if m["role"] == "assistant" and "analytics" in m]
        if ai_data_points:
            st.metric("Live Accuracy", f"{ai_data_points[-1].get('technical_accuracy', 0)}/10")
            st.line_chart(pd.DataFrame([d.get("technical_accuracy", 0) for d in ai_data_points], columns=["Score"]))

        if st.button("🛑 STOP & ANALYZE"):
            st.session_state.step = "evaluation"
            st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    if not st.session_state.chat_history:
        ai_data = safe_parse_json(st.session_state.agent.get_next_question(question_count=st.session_state.q_count))
        st.session_state.chat_history.append({"role": "assistant", "content": ai_data['question'], "analytics": ai_data})
        st.rerun()

    speak(st.session_state.chat_history[-1]["content"])

    user_text = None
    c_in, c_mic = st.columns([6, 1])
    with c_mic: 
        if st.button("🎤"): user_text = get_voice_input()
    with c_in: 
        chat_in = st.chat_input("Answer...")
        if chat_in: user_text = chat_in

    if user_text:
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        
        with st.spinner("Scoring..."):
            ai_data = safe_parse_json(st.session_state.agent.get_next_question(user_text, st.session_state.q_count))
            # ACCUMULATE SCORE (0.0 to 1.0)
            st.session_state.total_marks += float(ai_data.get('score', 0))
            st.session_state.chat_history.append({"role": "assistant", "content": ai_data['question'], "analytics": ai_data})
            st.session_state.q_count += 1
        
        if st.session_state.q_count > 20:
            st.session_state.step = "evaluation"
        st.rerun()

# 3. EVALUATION
elif st.session_state.step == "evaluation":
    st.markdown('<div class="step-tracker"><span class="step">1. PROFILE</span><span class="step">2. INTERVIEW</span><span class="step step-active">3. AUDIT</span></div>', unsafe_allow_html=True)
    st.title("🏆 Final Performance Audit")
    
    ai_msgs = [m for m in st.session_state.chat_history if m["role"] == "assistant" and "analytics" in m]
    
    if ai_msgs:
        final_stats = ai_msgs[-1]["analytics"]
        # Use .get to ensure safety
        final_mark = st.session_state.get("total_marks", 0.0)

        h1, h2, h3 = st.columns(3)
        with h1: st.markdown(f'<div class="metric-container"><h5>Final Grade</h5><h2>{final_mark:.1f}/20</h2></div>', unsafe_allow_html=True)
        with h2: st.markdown(f'<div class="metric-container"><h5>Mean Depth</h5><h2>{final_stats.get("depth")}/10</h2></div>', unsafe_allow_html=True)
        with h3: st.markdown(f'<div class="metric-container"><h5>Confidence</h5><h2>{final_stats.get("communication")}/10</h2></div>', unsafe_allow_html=True)

        st.markdown("---")
        tabs = st.tabs(["📋 Summary", "🧠 Breakdown", "🗺️ Roadmap"])

        with tabs[0]:
            transcript = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
            st.markdown(st.session_state.agent.get_feedback(transcript))
            
        with tabs[1]: 
            st.subheader("Technical Findings")
            st.write(f"**Strengths:** {final_stats.get('strengths')}")
            st.write(f"**Gaps:** {final_stats.get('improvements')}")
            st.info(f"**Interviewer Feedback:** {final_stats.get('feedback')}")

        with tabs[2]:
            st.success(f"### 🗺️ AI-Powered Learning Path: {st.session_state.matched_role}")
            
            st.subheader("📚 Recommended Specialized Courses")
            with st.spinner("Analyzing transcript for resources..."):
                ai_course_advice = st.session_state.agent.get_course_recommendations(transcript)
                st.markdown(ai_course_advice)

            st.markdown("---")
            st.subheader("🧠 Interactive Deep-Dive Tutor")
            st.write("Pick a recommended topic to start an end-to-end masterclass.")
            
            main_suggestion = final_stats.get('suggestion', 'Technical Fundamentals')
            selected_topic = st.text_input("Enter a topic to master:", value=main_suggestion)
            
            if st.button(f"📖 Start Masterclass on {selected_topic}"):
                tutor = TutorAgent()
                with st.spinner(f"Preparing tutorial for {selected_topic}..."):
                    explanation = tutor.tutor_on_recommendation(selected_topic, transcript)
                    st.markdown("---")
                    st.info(f"### 🎓 Tutor Session: {selected_topic}")
                    st.markdown(explanation)
            
            if st.button("🔄 NEW SESSION"):
                st.session_state.step = "setup"
                st.session_state.chat_history = []
                st.session_state.q_count = 1
                st.session_state.total_marks = 0.0
                st.rerun()
    else:
        st.error("No interview data available.")