import streamlit as st
import pandas as pd
import re
import pickle
from PyPDF2 import PdfReader

# Load the trained model and vectorizer
model_path = 'clf.pkl'  # Update with the actual path to your model file
vectorizer_path = 'tfidf.pkl'  # Update with the actual path to your vectorizer file

# Load the model and vectorizer
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the text cleaning function
def cleanResume(text):
    text = re.sub(r'http\S+\s', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'\#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Streamlit App Interface
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        font-family: Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #34495e;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2c3e50;
        font-size: 3em;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌟 Resume Screening App 🌟")

st.sidebar.header("📂 Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload resumes (text or PDF format)", accept_multiple_files=True, type=['txt', 'pdf'])

st.sidebar.info("👩‍💻 Developed for resume screening using machine learning.")

if uploaded_files:
    st.header("🔍 Predicted Categories")
    for file in uploaded_files:
        # Check file type and extract content
        if file.type == 'application/pdf':
            content = extract_text_from_pdf(file)
        else:
            content = file.read().decode('utf-8')

        # Clean and process the resume
        cleaned_content = cleanResume(content)

        # Vectorize the content
        vectorized_content = vectorizer.transform([cleaned_content]).toarray()

        # Predict the category
        prediction = model.predict(vectorized_content)
        st.success(f"**📄 {file.name}**: 🎯 {prediction[0]}")

st.markdown("""
    <footer style="text-align: center; font-size: 14px; padding-top: 20px;">
        Made with ❤️ using Streamlit
    </footer>
""", unsafe_allow_html=True)
