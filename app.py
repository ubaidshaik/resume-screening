import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import string
import nltk
import spacy
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download resources
nltk.download('stopwords')
spacy.cli.download("en_core_web_sm")

# Load NLP tools
nlp = spacy.load("en_core_web_sm")
stop_words = set(nltk.corpus.stopwords.words('english'))

# Function to extract text from different file types
def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # DOCX
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.type == "text/csv":
        df = pd.read_csv(file)
        text = " ".join(df.astype(str).values.flatten())  # Convert CSV data into text
    return text

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Function to compute similarity
def compute_similarity(resumes, job_description):
    """Compute TF-IDF and cosine similarity between job description and resumes."""
    vectorizer = TfidfVectorizer()
    documents = resumes + [job_description]  # Combine all resumes with the job description
    tfidf_matrix = vectorizer.fit_transform(documents)

    job_vector = tfidf_matrix[-1]  # Last element is the job description
    resume_vectors = tfidf_matrix[:-1]  # All other elements are resumes

    similarities = cosine_similarity(resume_vectors, job_vector).flatten()
    
    ranked_resumes = sorted(zip(resumes, similarities), key=lambda x: x[1], reverse=True)
    return ranked_resumes

# Streamlit App
st.title("📄 Automated Resume Screening System")

# File Upload Section
uploaded_files = st.file_uploader("Upload resumes (PDF, DOCX, CSV)", type=["pdf", "docx", "csv"], accept_multiple_files=True)

# Job Description Input
job_description = st.text_area("✍️ Enter Job Description")

if uploaded_files and job_description:
    resumes = []
    for file in uploaded_files:
        extracted_text = extract_text(file)
        cleaned_text = clean_text(extracted_text)
        resumes.append(cleaned_text)

    cleaned_job_description = clean_text(job_description)

    # Compute Similarity and Rank Resumes
    ranked_resumes = compute_similarity(resumes, cleaned_job_description)

    # Display Top 10 Matching Resumes
    st.subheader("🏆 Top 10 Matching Resumes")
    for idx, (resume_text, similarity) in enumerate(ranked_resumes[:10], start=1):
        st.write(f"**{idx}. Resume Similarity:** {similarity:.2f}")
        st.text_area(f"Resume {idx} Preview", resume_text[:500], height=150)  # Show first 500 chars of resume
