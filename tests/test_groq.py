import os
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# This forces Python to search your folders until it finds the .env file
load_dotenv(find_dotenv())

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ ERROR: Could not find GROQ_API_KEY. Please make sure your .env file exists and contains the key.")
    exit()

print("✅ API Key found! Sending request to Groq...")

try:
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Respond with the word SUCCESS"}],
        model="llama-3.3-70b-versatile",
    )
    print("Groq Response:", chat_completion.choices[0].message.content)
except Exception as e:
    print("Error connecting to Groq:", e)