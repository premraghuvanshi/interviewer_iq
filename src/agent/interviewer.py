import os
import json
from dotenv import load_dotenv, find_dotenv
from groq import Groq
from textblob import TextBlob

load_dotenv(find_dotenv())

def get_prompt(role, question_count):
    """
    Generates a dynamic prompt that increases in difficulty based on the question count.
    """
    if question_count <= 5:
        difficulty = "EASY: Focus on fundamental syntax, core definitions, and basic concepts."
    elif question_count <= 12:
        difficulty = "INTERMEDIATE: Focus on practical application, libraries, and logic."
    elif question_count <= 18:
        difficulty = "ADVANCED: Focus on system architecture, optimization, and complex problem-solving."
    else:
        difficulty = "EXPERT: Focus on edge cases, deep theoretical trade-offs, and high-level strategy."

    return f"""
You are a STRICT and REALISTIC technical interviewer for a {role} position.

Current Progress: Question {question_count} of 20.
Difficulty Level: {difficulty}

Your behavior:
- Ask exactly ONE question at a time.
- Evaluate the candidate's last response with high scrutiny.
- Detect vague or memorized answers.
- If an answer is weak → ask a challenging follow-up.
- If an answer is strong → move to a more complex concept within the {difficulty} range.

Evaluation Criteria (score 0–10):
1. Technical Accuracy
2. Depth of Knowledge
3. Communication Clarity
4. Confidence

Rules:
- Respond ONLY in VALID JSON format.
- Do NOT include any conversational text before or after the JSON.
- DO NOT use markdown code blocks (like ```json).
- Penalize incorrect or incomplete explanations.

{{
  "question": "the next interview question",
  "score": number,
  "technical_accuracy": number,
  "depth": number,
  "communication": number,
  "confidence": number,
  "strengths": "brief analysis of what they did well",
  "improvements": "specific technical gaps identified",
  "feedback": "direct, professional interviewer feedback",
  "follow_up": "a deeper dive question if needed, otherwise empty",
  "suggestion": "specific learning resources or topics to master"
}}
"""

class InterviewAgent:
    def __init__(self, role):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.role = role
        self.model = "llama-3.3-70b-versatile"
        self.messages = []

    def get_next_question(self, user_input=None, question_count=1):
        """
        Fetches the next question with a multi-layered JSON safety shield.
        """
        system_msg = {"role": "system", "content": get_prompt(self.role, question_count)}
        
        if not self.messages:
            self.messages.append(system_msg)
        else:
            self.messages[0] = system_msg 

        if user_input:
            self.messages.append({"role": "user", "content": user_input})
        else:
            self.messages.append({"role": "user", "content": "I am ready. Please start the interview."})

        try:
            # SHIELD 1: Force JSON mode at the API level
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                response_format={"type": "json_object"}
            )
            
            content = completion.choices[0].message.content
            
            # SHIELD 2: Robust internal parsing
            ai_data = json.loads(content)
            
            # Record assistant history
            self.messages.append({"role": "assistant", "content": ai_data.get('question', '')})
            
            return ai_data
            
        except Exception as e:
            # SHIELD 3: The "Emergency Fallback" dictionary
            return {
                "question": "I apologize, I encountered a technical sync issue. Could you please repeat your last technical point?",
                "score": 0,
                "technical_accuracy": 0,
                "depth": 0,
                "communication": 0,
                "confidence": 0,
                "strengths": "Connection fluctuation",
                "improvements": "N/A",
                "feedback": f"System Sync Error: {str(e)}",
                "follow_up": "",
                "suggestion": "Please ensure your internet connection is stable."
            }

    def get_feedback(self, transcript):
        """Generates final report summary."""
        prompt = f"""
        Analyze this interview transcript for a {self.role} position.
        Summarize:
        1. Overall Technical Proficiency
        2. Soft Skills & Confidence
        3. Final Recommendation (Hire/No Hire) with justification.
        
        Transcript:
        {transcript}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Feedback Error: {e}"

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0.1:
            return "Positive / Confident"
        elif analysis.sentiment.polarity < -0.1:
            return "Negative / Hesitant"
        else:
            return "Neutral / Formal"
    def get_course_recommendations(self, transcript):
        """
        Uses Groq to suggest specific courses with clickable URLs based on identified gaps.
        """
        prompt = f"""
        Analyze this interview transcript for a {self.role} position:
        {transcript}
        
        Identify the top 3 technical gaps. For each gap:
        1. Suggest one specific, high-quality course or certification.
        2. Provide a valid, clickable markdown URL to the course (e.g., Coursera, NPTEL, or YouTube).
        3. Explain briefly why this course helps bridge that specific gap.

        Respond ONLY in a clean, professional bullet-point format. Ensure the links are functional.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception:
            return "Unable to fetch specialized courses. Please check your internet connection."