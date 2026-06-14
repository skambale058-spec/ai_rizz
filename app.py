import streamlit as st
import fitz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO

# ---------------- UI ----------------
st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    layout="wide"
)

st.title("🚀 AI Resume Analyzer Pro")

# ---------------- SKILLS DB ----------------
SKILLS_DB = [
    "python","java","javascript","react","nodejs","django","flask",
    "fastapi","aws","azure","gcp","docker","kubernetes",
    "mongodb","postgresql","mysql","redis",
    "machine learning","deep learning","nlp",
    "html","css","git","linux"
]

# ---------------- PDF PARSER ----------------
def parse_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text):
    text = text.lower()
    return list(set([s for s in SKILLS_DB if s in text]))

# ---------------- JD ANALYSIS ----------------
def jd_analysis(jd_text):
    return extract_skills(jd_text)

# ---------------- MATCH SCORE ----------------
def match_score(resume, jd):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume, jd])
    return round(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100, 2)

# ---------------- EXPERIENCE ----------------
def experience(text):
    years = re.findall(r"(20\d{2}|19\d{2})", text)
    if len(years) < 2:
        return 0
    years = sorted([int(y) for y in years])
    return max(0, years[-1] - years[0])

# ---------------- RECOMMENDATIONS ----------------
def recommendations(score, missing, exp):
    tips = []

    if score < 70:
        tips.append("Improve resume structure and add keywords from JD")

    if missing:
        tips.append("Add missing skills: " + ", ".join(missing))

    if exp < 2:
        tips.append("Highlight projects and internships")

    if score > 85:
        tips.append("Excellent profile — ready for interviews")

    return tips

# ---------------- INTERVIEW QUESTIONS ----------------
def interview_questions(skills):
    questions = []

    if "python" in skills:
        questions.append("Explain Python memory management")

    if "django" in skills:
        questions.append("What is Django ORM?")

    if "react" in skills:
        questions.append("Difference between state and props")

    if "machine learning" in skills:
        questions.append("Explain overfitting and underfitting")

    if not questions:
        questions.append("Tell me about your last project")

    return questions

# ---------------- EXPORT ----------------
def create_report(resume_text, jd_text, score, skills, missing, exp):
    report = f"""
AI RESUME ANALYSIS REPORT
-------------------------

MATCH SCORE: {score}%

EXPERIENCE: {exp} years

RESUME SKILLS:
{skills}

MISSING SKILLS:
{missing}

RECOMMENDATIONS:
{recommendations(score, missing, exp)}

JD ANALYSIS:
{jd_analysis(jd_text)}

INTERVIEW QUESTIONS:
{interview_questions(skills)}

RESUME TEXT (PREVIEW):
{resume_text[:2000]}
"""
    return report

# ---------------- INPUT ----------------
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description", height=200)

# ---------------- RUN ----------------
if st.button("Analyze Resume 🚀"):

    if not resume_file or not jd_text.strip():
        st.error("Please upload resume and JD")
        st.stop()

    resume_text = parse_pdf(resume_file)

    resume_skills = extract_skills(resume_text)
    jd_skills = jd_analysis(jd_text)

    score = match_score(resume_text, jd_text)

    missing = list(set(jd_skills) - set(resume_skills))

    exp = experience(resume_text)

    tips = recommendations(score, missing, exp)

    questions = interview_questions(resume_skills)

    # ---------------- UI OUTPUT ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Match Score", f"{score}%")
    col2.metric("Experience", f"{exp} yrs")
    col3.metric("Skills Found", len(resume_skills))

    st.subheader("🧠 Parsed Resume Skills")
    st.write(resume_skills)

    st.subheader("❌ Missing Skills")
    st.write(missing)

    st.subheader("💡 Recommendations")
    for t in tips:
        st.success(t)

    st.subheader("🎯 Interview Questions")
    for q in questions:
        st.info(q)

    st.subheader("📄 Resume Preview")
    st.text(resume_text[:3000])

    # ---------------- DOWNLOAD ----------------
    report = create_report(
        resume_text,
        jd_text,
        score,
        resume_skills,
        missing,
        exp
    )

    st.download_button(
        "📥 Download Report",
        report,
        file_name="resume_analysis.txt"
    )
