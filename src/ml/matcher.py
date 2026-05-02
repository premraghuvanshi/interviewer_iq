from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RoleMatcher:
    def __init__(self):
        """
        Initializes the RoleMatcher with a pre-defined technical knowledge base.
        The dictionary maps professional roles to their core technical stacks.
        """
        # Expanded skill sets based on industry standards and your project requirements
        self.role_data = {
            "Data Analyst": "python sql my sql excel power bi data visualization analytics tableau dashboard",
            "Machine Learning Engineer": "python machine learning ml deep learning tensorflow pytorch scikit-learn neural networks",
            "Web Developer": "html css javascript react node js web development frontend backend stack",
            "Backend Developer": "python django flask api database sql postgresql mongodb backend server restful",
            "Data Scientist": "python statistics machine learning pandas numpy science analytics scikit-learn modeling",
            "Frontend Developer": "html css javascript react vue ui/ux ui ux frontend design tailwind bootstrap",
            "Full Stack Developer": "html css javascript react node mongodb express fullstack web stack sql",
            "DevOps Engineer": "docker kubernetes ci/cd ci cd aws linux automation jenkins terraform ansible",
            "Cloud Engineer": "aws azure gcp cloud architecture networking infrastructure terraform cloud-computing",
            "AI Engineer": "python machine learning nlp deep learning llms artificial intelligence transformers generative-ai"
        }
        
        self.roles = list(self.role_data.keys())
        self.skill_documents = list(self.role_data.values())
        
        # Stop_words='english' removes common non-technical words like 'am', 'is', 'the'
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.skill_documents)

    def predict_role(self, user_skills: str) -> str:
        """
        Analyzes candidate input and returns the most relevant job role.
        
        Args:
            user_skills (str): The raw text input from the candidate's profile.
            
        Returns:
            str: The predicted professional role or 'General Candidate'.
        """
        # Handle empty or whitespace-only input
        if not user_skills or not user_skills.strip():
            return "General Candidate"

        # Preprocess and vectorize the candidate's input
        user_vector = self.vectorizer.transform([user_skills.lower()])
        
        # Calculate Cosine Similarity against all roles in the database
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)
        max_sim = np.max(similarities)
        
        # Logic Gate: If the overlap is too low (less than 15%), we don't assume a role
        if max_sim < 0.15:
            return "General Candidate"
            
        # Select the role index with the highest similarity score
        best_match_index = np.argmax(similarities)
        return self.roles[best_match_index]

if __name__ == "__main__":
    # Internal Unit Test for Team Verification
    matcher = RoleMatcher()
    
    test_inputs = [
        "I am a 3rd year student working on LLMs and NLP projects using Python.",
        "Experienced in React, HTML, and CSS for building responsive web interfaces.",
        "I manage Kubernetes clusters and automated CI/CD pipelines on AWS.",
        "I just know some basic Python and like solving math problems."
    ]
    
    print("--- Interviewer IQ: ML Matcher Unit Test ---")
    for text in test_inputs:
        prediction = matcher.predict_role(text)
        print(f"Input: '{text[:60]}...' \nResult: {prediction}\n")