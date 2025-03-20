import streamlit as st
import pandas as pd
import numpy as np
import nltk
import spacy
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load NLP model
nlp = spacy.load("en_core_web_sm")
print("Model loaded successfully!")

def clean_text(text):
    """Function to clean resume text by removing punctuation, numbers, and stopwords."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def extract_features(resume_data, job_description):
    """Extract TF-IDF features and compute similarity scores."""
    vectorizer = TfidfVectorizer()
    all_texts = resume_data['Cleaned_Resume'].tolist() + [clean_text(job_description)]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    job_vector = tfidf_matrix[-1]  # Last entry is the job description
    resume_vectors = tfidf_matrix[:-1]  # All other entries are resumes

    # Compute cosine similarity
    similarity_scores = cosine_similarity(resume_vectors, job_vector)
    resume_data['Similarity'] = similarity_scores.flatten()

    return resume_data.sort_values(by='Similarity', ascending=False)

# Streamlit Web App
st.title("Automated Resume Screening System")

# Upload Resume Dataset
uploaded_file = st.file_uploader("Upload CSV file containing resumes", type=["csv"])

# Job Description Input
job_description = st.text_area("Enter Job Description")
if uploaded_file is not None:
    try:
        resume_data = pd.read_csv(uploaded_file)
        st.write("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading the file: {e}")
else:
    st.warning("Please upload a valid CSV file.")
if uploaded_file and job_description:
    # Load Resume Data
    resume_data = pd.read_csv(uploaded_file)

    # Preprocess Resumes
    resume_data['Cleaned_Resume'] = resume_data['Resume'].apply(clean_text)

    # Extract Features and Rank Resumes
    ranked_resumes = extract_features(resume_data, job_description)

    # Display Results
    st.subheader("Top 10 Matching Resumes")
    st.write(ranked_resumes[['Resume', 'Similarity']].head(10))
