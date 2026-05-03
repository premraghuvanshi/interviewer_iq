⚖️ InterviewIQ
AI-Powered Technical Viva & Career Development Agent
Developed by: Prem Raghuvanshi

📌 Project Overview
Interviewer IQ is a sophisticated automated assessment platform designed to conduct professional technical vivas. It utilizes a Dual-Agent Architecture to transition seamlessly from a high-pressure interviewer to a supportive technical tutor. The system is specifically optimized for engineering students to test their proficiency in roles like Machine Learning Engineer, Data Scientist, and AI Engineer.

🚀 Key Features
Dynamic Role Matching: Uses a custom ML RoleMatcher to analyze candidate backgrounds and assign appropriate interview tracks.

Four-Tier Difficulty Scaling: Questions evolve from basic concepts (Tier 1) to expert-level architecture and deep mechanics (Tier 4) based on progress.

Advanced Voice Interaction:

5-Second Silence Detection: Implements natural-feel Voice Activity Detection (VAD) that waits for 5 seconds of silence before processing audio, allowing for thoughtful technical responses.

Text-to-Speech (TTS): Real-time audio generation for interviewer questions using gTTS.

Dual-Agent System:

Interviewer Agent: Focuses on rigorous evaluation and weighted scoring out of 20 marks.

Tutor Agent: Provides an "End-to-End" masterclass on identified knowledge gaps after the session.

Security Shielding: Built-in protection against prompt injection to ensure grading integrity.

🛠️ Tech Stack
Language: Python 3.10+

LLM Inference: Groq (Llama 3.3 70B Versatile)

Frontend: Streamlit

NLP: TextBlob (Sentiment & Confidence Analysis)

Audio: SpeechRecognition (Google API) & gTTS

📂 Project Structure
Plaintext
ai_interviewer_project/
├── app.py                # Main Streamlit Application
├── .env                  # Environment Variables (API Keys)
├── src/
│   ├── agent/
│   │   └── interviewer.py # Agent Logic (Interviewer & Tutor)
│   └── ml/
│       └── matcher.py     # Role Matching Logic
└── requirements.txt      # Dependencies
⚙️ Installation & Usage
Clone the repository:

Bash
git clone https://github.com/premraghuvanshi/interviewer_iq.git
cd interviewer-iq
Install requirements:

Bash
pip install -r requirements.txt
python -m textblob.download_corpora
Set up your .env file:

Code snippet
GROQ_API_KEY=your_api_key_here
Run the app:

Bash
streamlit run app.py
