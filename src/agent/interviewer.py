import os
from dotenv import load_dotenv, find_dotenv
from groq import Groq

load_dotenv(find_dotenv())

class InterviewAgent:
    def __init__(self, role):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.role = role
        self.model = "llama-3.3-70b-versatile"
        self.messages = [
            {
                "role": "system",
                "content": f"""You are an expert technical interviewer for a {self.role} position. 
                Your goal is to conduct a professional, concise interview.
                1. Ask only ONE technical question at a time.
                2. If the user's answer is brief, ask a follow-up cross-question.
                3. Stay in character. Do not give feedback until the end.
                4. Keep your responses under 50 words."""
            }
        ]

    def get_next_question(self, user_input=None):
        if user_input:
            self.messages.append({"role": "user", "content": user_input})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            ai_response = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": ai_response})
            return ai_response
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    # Quick Test: Simulated Data Science Interview
    print(f"--- Starting Test Interview for Data Scientist ---")
    agent = InterviewAgent("Data Scientist")
    
    # Get initial question
    first_q = agent.get_next_question()
    print(f"AI: {first_q}")
    
    # Simulate a user answer
    user_ans = "I prefer using Random Forest for classification tasks."
    print(f"User: {user_ans}")
    
    follow_up = agent.get_next_question(user_ans)
    print(f"AI (Follow-up): {follow_up}")