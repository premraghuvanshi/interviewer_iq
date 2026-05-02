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
    # Logic for increasing technical difficulty
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
- Be analytical and slightly critical; detect vague or memorized answers.
- If an answer is weak → ask a challenging follow-up.
- If an answer is strong → move to a more complex concept within the {difficulty} range.

Evaluation Criteria (score 0–10):
1. Technical Accuracy
2. Depth of Knowledge
3. Communication Clarity
4. Confidence

Rules:
- Respond ONLY in VALID JSON format.
- DO NOT be overly nice or supportive.
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
        Fetches the next question while adjusting difficulty dynamically.
        """
        # Dynamically refresh the system prompt with the current difficulty level
        system_msg = {"role": "system", "content": get_prompt(self.role, question_count)}
        
        # Reset or update messages to include the new difficulty context
        if not self.messages:
            self.messages.append(system_msg)
        else:
            self.messages[0] = system_msg # Update the first message (system prompt)

        if user_input:
            self.messages.append({"role": "user", "content": user_input})
        else:
            self.messages.append({"role": "user", "content": "I am ready. Please start the interview."})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                response_format={"type": "json_object"}
            )
            
            ai_data = json.loads(completion.choices[0].message.content)
            
            # Record the question in history to maintain context
            self.messages.append({"role": "assistant", "content": ai_data['question']})
            
            return ai_data
            
        except Exception as e:
            return {
                "error": str(e), 
                "question": "System sync error. Could you please repeat your last technical point?",
                "technical_accuracy": 0, "depth": 0, "communication": 0, "confidence": 0,
                "strengths": "N/A", "improvements": "N/A", "feedback": "Error", "suggestion": "N/A"
            }

    def get_feedback(self, transcript):
        """
        Generates final high-level feedback after the interview ends.
        """
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

if __name__ == "__main__":
    # Test for an AI Engineer role at question 15 (Advanced difficulty)
    agent = InterviewAgent("AI Engineer")
    print(agent.get_next_question(question_count=15))