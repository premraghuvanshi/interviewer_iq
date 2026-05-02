from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RoleMatcher:
    def __init__(self):
        # Expanded skill sets with common variations to improve matching accuracy
        # This acts as our 'training data' for the TF-IDF model
        self.role_data = {
            "Data Scientist": "python machine learning ml deep learning pandas numpy scikit-learn sklearn data science analytics ai artificial intelligence nlp computer vision stats statistics",
            "Frontend Developer": "html css javascript react vue js frontend ui ux tailwind bootstrap web development typescript angular sass web-design nextjs",
            "Backend Developer": "python django flask java spring boot nodejs backend server sql database mongodb api restful microservices golang php ruby",
            "Data Analyst": "sql excel powerbi tableau dashboard data visualization statistics analytics reporting business intelligence looker cleaning preprocessing",
            "DevOps Engineer": "linux docker kubernetes aws cloud devops jenkins git ci cd automation azure gcp terraform ansible monitoring"
        }
        
        self.roles = list(self.role_data.keys())
        self.skill_documents = list(self.role_data.values())
        
        # stop_words='english' filters out 'i', 'know', 'am', 'the' 
        # so only technical keywords are compared
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.skill_documents)

    def predict_role(self, user_skills: str) -> str:
        """
        Takes user input and finds the closest matching job role using Cosine Similarity.
        """
        # Return default if input is empty or just whitespace
        if not user_skills or not user_skills.strip():
            return "General Candidate"

        # Preprocess: lowercase the input and vectorize
        user_vector = self.vectorizer.transform([user_skills.lower()])
        
        # Calculate similarity scores between input and all role documents
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)
        max_sim = np.max(similarities)
        
        # Threshold Logic:
        # If the highest similarity score is less than 0.1, we don't have enough 
        # evidence to pick a specific role, so we return 'General Candidate'.
        if max_sim < 0.1:
            return "General Candidate"
            
        # Otherwise, pick the role with the highest score
        best_match_index = np.argmax(similarities)
        return self.roles[best_match_index]

if __name__ == "__main__":
    # Internal Unit Test
    matcher = RoleMatcher()
    
    # Test cases to verify the fix
    test_cases = [
        "I know python", 
        "I am good at React and CSS",
        "I use SQL and PowerBI",
        "Just a general student"
    ]
    
    print("--- Testing ML Role Matcher ---")
    for text in test_cases:
        prediction = matcher.predict_role(text)
        print(f"Input: '{text}' -> Predicted: {prediction}")