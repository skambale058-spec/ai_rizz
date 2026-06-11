import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Screening System")

job_desc = st.text_area("Enter Job Description")
resume = st.text_area("Paste Resume")

if st.button("Analyze"):

    if job_desc.strip() == "" or resume.strip() == "":
        st.error("Please enter both Job Description and Resume.")
    else:
        docs = [job_desc, resume]

        tfidf = TfidfVectorizer(stop_words="english")
        matrix = tfidf.fit_transform(docs)

        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

        st.success(f"Match Score: {round(score * 100, 2)}%")
