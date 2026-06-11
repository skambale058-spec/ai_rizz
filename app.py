import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.title("AI Resume Screening System (PDF Upload)")

job_desc = st.text_area("Enter Job Description")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

if st.button("Analyze"):

    if job_desc.strip() == "" or uploaded_file is None:
        st.error("Please enter Job Description and upload a PDF resume.")
    else:
        resume_text = extract_text_from_pdf(uploaded_file)

        docs = [job_desc, resume_text]

        tfidf = TfidfVectorizer(stop_words="english")
        matrix = tfidf.fit_transform(docs)

        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

        st.success(f"Match Score: {round(score * 100, 2)}%")

        st.subheader("Extracted Resume Text")
        st.write(resume_text)
