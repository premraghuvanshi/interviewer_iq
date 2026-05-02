import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RoleMatcher:
    def __init__(self):
        # Your specific fields preserved for exact matching logic
        self.roles_data = {
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
        self.vectorizer = TfidfVectorizer()
        
    def predict_role(self, user_skills):
        """
        Maps user skills to the closest predefined role. 
        Defaults to 'General Candidate' if no strong match is found.
        """
        # 1. Check for empty or extremely short input
        if not user_skills or len(user_skills.strip()) < 3:
            return "General Candidate"

        # 2. Prepare role profiles for comparison
        role_names = list(self.roles_data.keys())
        role_keywords = list(self.roles_data.values())
        
        # 3. Combine role keywords with user input
        # We append the user_skills at the end of the list
        comparison_set = role_keywords + [user_skills.lower()]
        
        # 4. Generate TF-IDF Matrix
        # This converts text into numerical vectors based on word importance
        tfidf_matrix = self.vectorizer.fit_transform(comparison_set)
        
        # 5. Calculate Cosine Similarity
        # Compare the user's vector (last index) against the role vectors (all previous)
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        
        # 6. Find the best match
        best_match_index = similarities.argmax()
        max_similarity = similarities[0][best_match_index]
        
        # 7. LOGIC: If similarity is too low (below 0.1), it's a "General Candidate"
        # This prevents the AI from forcing a user into a specific role they don't fit.
        if max_similarity > 0.1:
            return role_names[best_match_index]
            
        return "General Candidate"