import streamlit as st
import re
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer Pro", layout="wide")

st.title("🚀 AI Resume Analyzer (Professional)")

SKILLS_DB = [
    "python","java","javascript","react","nodejs","django","flask",
    "fastapi","aws","azure","gcp","docker","kubernetes",
    "mongodb","postgresql","mysql","redis",
    "machine learning","deep learning","nlp",
    "html","css","git","linux"
]

def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_skills(text):
    text = text.lower()
    return list(set([s for s in SKILLS_DB if s in text]))

def semantic_score(resume, jd):
    tfidf = TfidfVectorizer()
    v = tfidf.fit_transform([resume, jd])
    return round(cosine_similarity(v[0:1], v[1:2])[0][0] * 100, 2)

def experience(text):
    years = re.findall(r"(20\d{2}|19\d{2})", text)
    if len(years) < 2:
        return 0
    years = sorted([int(y) for y in years])
    return max(0, years[-1] - years[0])

def format_score(text):
    score = 100
    if "experience" not in text.lower(): score -= 20
    if "education" not in text.lower(): score -= 20
    if len(text) < 500: score -= 20
    return max(score, 0)

def ats(sem, skill, exp, fmt):
    return round(sem*0.4 + skill*0.3 + exp*0.2 + fmt*0.1, 2)

resume = st.file_uploader("Upload Resume PDF", type=["pdf"])
jd = st.text_area("Paste Job Description")

if st.button("Analyze"):

    if resume is None or jd.strip() == "":
        st.error("Upload resume and JD first")
        st.stop()

    text = extract_text(resume)

    r_skills = extract_skills(text)
    j_skills = extract_skills(jd)

    sem = semantic_score(text, jd)

    skill_match = 0
    if j_skills:
        skill_match = len(set(r_skills) & set(j_skills)) / len(j_skills) * 100

    exp = experience(text)
    fmt = format_score(text)

    score = ats(sem, skill_match, exp, fmt)

    missing = list(set(j_skills) - set(r_skills))

    st.metric("ATS Score", f"{score}%")
    st.write("Semantic:", sem)
    st.write("Skills Found:", r_skills)
    st.write("Missing Skills:", missing)
    st.write("Experience:", exp, "years")
