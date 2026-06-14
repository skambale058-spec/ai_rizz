import streamlit as st
import re
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    page_icon="📄",
    layout="wide"
)

# ---------------- PROFESSIONAL UI ----------------
st.markdown("""
<style>
    .main {
        background-color: #0b1220;
        color: white;
    }
    .stApp {
        background: linear-gradient(to right, #0f172a, #111827);
    }
    h1, h2, h3 {
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SKILLS DATABASE ----------------
SKILLS_DB = [
    "python","java","javascript","react","nodejs","django","flask",
    "fastapi","aws","azure","gcp","docker","kubernetes",
    "mongodb","postgresql","mysql","redis",
    "machine learning","deep learning","nlp",
    "git","linux","html","css","typescript"
]

# ---------------- PDF READER ----------------
def extract_text(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text):
    text = text.lower()
    return list(set([skill for skill in SKILLS_DB if skill in text]))

# ---------------- EXPERIENCE CALCULATION ----------------
def calculate_experience(text):
    years = re.findall(r"(20\d{2}|19\d{2})", text)
    if len(years) < 2:
        return 0
    years = sorted([int(y) for y in years])
    return max(0, years[-1] - years[0])

# ---------------- SEMANTIC MATCH ----------------
def semantic_match(resume, jd):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume, jd])
    return round(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100, 2)

# ---------------- FORMAT SCORE ----------------
def format_score(text):
    score = 100
    if "experience" not in text.lower():
        score -= 15
    if "education" not in text.lower():
        score -= 15
    if "skills" not in text.lower():
        score -= 15
    if len(text) < 500:
        score -= 20
    return max(score, 0)

# ---------------- ATS SCORE ----------------
def ats_score(semantic, skill, exp, fmt):
    return round(
        semantic * 0.40 +
        skill * 0.30 +
        fmt * 0.20 +
        exp * 0.10,
        2
    )

# ---------------- FEEDBACK ENGINE ----------------
def feedback(score, missing, exp):
    tips = []

    if score < 70:
        tips.append("Improve resume structure and keywords.")

    if missing:
        tips.append("Add missing skills: " + ", ".join(missing))

    if exp < 2:
        tips.append("Add internships and real projects.")

    if score >= 85:
        tips.append("Excellent resume quality!")

    return tips

# ---------------- HEADER ----------------
st.title("🚀 Professional AI Resume Analyzer")
st.caption("ATS Score + Skill Gap + AI Matching Engine")

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_text = st.text_area("Paste Job Description", height=250)

# ---------------- ANALYZE BUTTON ----------------
if st.button("Analyze Resume 🚀"):

    if not resume_file or not jd_text.strip():
        st.error("Please upload resume and job description")
        st.stop()

    resume_text = extract_text(resume_file)

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    semantic = semantic_match(resume_text, jd_text)

    skill_match = 0
    if jd_skills:
        skill_match = len(set(resume_skills) & set(jd_skills)) / len(jd_skills) * 100

    missing = list(set(jd_skills) - set(resume_skills))

    exp = calculate_experience(resume_text)

    fmt = format_score(resume_text)

    final = ats_score(semantic, skill_match, exp, fmt)

    tips = feedback(final, missing, exp)

    # ---------------- DASHBOARD ----------------
    st.subheader("📊 ATS Score Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("ATS Score", f"{final}%")
    c2.metric("Semantic Match", f"{semantic}%")
    c3.metric("Skill Match", f"{round(skill_match,2)}%")
    c4.metric("Experience", f"{exp} yrs")

    st.divider()

    # ---------------- DETAILS ----------------
    st.subheader("🧠 Skills Found")
    st.write(resume_skills)

    st.subheader("❌ Missing Skills")
    st.write(missing)

    st.subheader("💡 AI Recommendations")
    for t in tips:
        st.success(t)

    st.subheader("📄 Resume Preview")
    st.text_area("", resume_text[:4000], height=300)
