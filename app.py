import streamlit as st
import fitz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    .stApp {
        background-color: #0f172a;
    }
    h1, h2, h3 {
        color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SKILLS ----------------
SKILLS_DB = [
    "python","java","javascript","react","nodejs","django",
    "fastapi","flask","aws","azure","gcp","docker","kubernetes",
    "mongodb","postgresql","mysql","redis","machine learning",
    "deep learning","nlp","git","linux","html","css","typescript"
]

# ---------------- PDF READER ----------------
def read_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text):
    text = text.lower()
    return list(set([s for s in SKILLS_DB if s in text]))

# ---------------- EXPERIENCE ----------------
def get_experience(text):
    years = re.findall(r"(20\d{2}|19\d{2})", text)
    if len(years) < 2:
        return 0
    years = sorted([int(y) for y in years])
    return max(0, years[-1] - years[0])

# ---------------- SEMANTIC MATCH ----------------
def semantic_score(resume, jd):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, jd])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

# ---------------- ATS SCORE ----------------
def ats_score(semantic, skill_match, exp_score, format_score):
    return round(
        semantic * 0.4 +
        skill_match * 0.3 +
        exp_score * 0.2 +
        format_score * 0.1,
        2
    )

# ---------------- FORMAT CHECK ----------------
def format_score(text):
    score = 100
    if "experience" not in text.lower():
        score -= 20
    if "education" not in text.lower():
        score -= 20
    if len(text) < 500:
        score -= 20
    return max(score, 0)

# ---------------- UI ----------------
st.title("🚀 AI Resume Analyzer Pro (ATS + AI Matching)")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_text = st.text_area("🧾 Paste Job Description")

if st.button("Analyze Resume 🚀"):

    if resume_file is None:
        st.error("Please upload resume")
        st.stop()

    if jd_text.strip() == "":
        st.error("Please enter job description")
        st.stop()

    # extract text
    resume_text = read_pdf(resume_file)

    # skills
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    # matching
    semantic = semantic_score(resume_text, jd_text)

    skill_match = 0
    if jd_skills:
        skill_match = len(set(resume_skills) & set(jd_skills)) / len(jd_skills) * 100

    missing_skills = list(set(jd_skills) - set(resume_skills))

    exp = get_experience(resume_text)

    fmt = format_score(resume_text)

    final_score = ats_score(semantic, skill_match, exp, fmt)

    # ---------------- OUTPUT ----------------
    st.subheader("📊 ATS Score")
    st.metric("Final Score", f"{final_score}%")

    c1, c2, c3 = st.columns(3)

    c1.metric("Semantic Match", f"{semantic}%")
    c2.metric("Skill Match", f"{round(skill_match,2)}%")
    c3.metric("Experience (yrs)", exp)

    st.subheader("🧠 Detected Skills")
    st.write(resume_skills)

    st.subheader("❌ Missing Skills")
    st.write(missing_skills)

    st.subheader("📌 Recommendations")

    if final_score < 70:
        st.warning("Improve resume structure + add missing skills")

    if missing_skills:
        st.info("Add these skills: " + ", ".join(missing_skills))

    if exp < 2:
        st.info("Add more projects/internships to improve profile")

    if final_score > 80:
        st.success("Strong resume! Good job 👍")

    st.subheader("📄 Resume Preview")
    st.text(resume_text[:3000])
