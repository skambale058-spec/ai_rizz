import streamlit as st
import fitz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Career Super Suite", layout="wide")

st.title("🚀 AI Career Super Suite (All-in-One Platform)")

# ---------------- SKILL DATABASE ----------------
SKILLS = [
    "python","java","javascript","react","nodejs","django","flask",
    "fastapi","aws","docker","kubernetes","sql",
    "machine learning","deep learning","nlp",
    "html","css","git","linux"
]

JOBS = {
    "Backend Developer": "python django fastapi sql docker",
    "Frontend Developer": "html css javascript react ui",
    "Data Scientist": "python machine learning deep learning sql",
    "DevOps Engineer": "aws docker kubernetes linux ci cd"
}

# ---------------- PDF PARSER ----------------
def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ---------------- SKILLS ----------------
def extract_skills(text):
    text = text.lower()
    return list(set([s for s in SKILLS if s in text]))

# ---------------- MATCH SCORE ----------------
def score(resume, jd):
    tfidf = TfidfVectorizer()
    v = tfidf.fit_transform([resume, jd])
    return round(cosine_similarity(v[0], v[1])[0][0] * 100, 2)

# ---------------- EXPERIENCE ----------------
def experience(text):
    years = re.findall(r"(20\d{2}|19\d{2})", text)
    if len(years) < 2:
        return 0
    years = sorted([int(y) for y in years])
    return max(0, years[-1] - years[0])

# ---------------- INPUT ----------------
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

jd_text = st.text_area("Paste Job Description (Optional)")

# ---------------- RUN ----------------
if st.button("Analyze Career 🚀"):

    if not resume_file:
        st.warning("Upload resume first")
        st.stop()

    resume_text = extract_text(resume_file)

    skills = extract_skills(resume_text)

    exp = experience(resume_text)

    # ---------------- JOB MATCHING ----------------
    job_results = []

    for job, desc in JOBS.items():
        job_results.append((job, score(resume_text, desc)))

    job_results.sort(key=lambda x: x[1], reverse=True)

    best_job = job_results[0][0]

    # ---------------- OUTPUT ----------------
    st.subheader("📊 Resume Insights")

    st.write("Skills:", skills)
    st.write("Experience:", exp, "years")

    st.subheader("💼 Best Job Matches")

    for j, s in job_results:
        st.write(f"{j} → {s}% fit")

    st.success(f"Best Recommended Role: {best_job}")

    # ---------------- INTERVIEW QUESTIONS ----------------
    st.subheader("🎤 Interview Questions")

    QUESTIONS = {
        "Backend Developer": ["What is API?", "Explain database indexing", "What is microservices?"],
        "Frontend Developer": ["What is React?", "DOM explanation", "State vs props"],
        "Data Scientist": ["What is ML?", "Overfitting?", "Model training process"],
        "DevOps Engineer": ["What is Docker?", "CI/CD?", "Kubernetes basics?"]
    }

    for q in QUESTIONS.get(best_job, []):
        st.write("•", q)

    # ---------------- CAREER ADVICE ----------------
    st.subheader("💡 Career Advice")

    top_score = job_results[0][1]

    if top_score > 80:
        st.success("You are job ready 🚀")
    elif top_score > 50:
        st.info("Keep improving skills")
    else:
        st.warning("Focus on fundamentals first")
