```python
import streamlit as st
import fitz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# --------------------------
# SKILLS DATABASE
# --------------------------
SKILLS = [
    "python","java","javascript","typescript",
    "react","nextjs","nodejs","express",
    "fastapi","django","flask",
    "aws","azure","gcp",
    "docker","kubernetes",
    "mongodb","postgresql","mysql",
    "redis","tensorflow","pytorch",
    "machine learning","deep learning",
    "git","linux","microservices",
    "data science","sql","html","css"
]

# --------------------------
# PDF PARSER
# --------------------------
def extract_pdf_text(uploaded_file):
    text = ""

    pdf = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    for page in pdf:
        text += page.get_text()

    return text

# --------------------------
# SKILL EXTRACTION
# --------------------------
def extract_skills(text):
    text = text.lower()

    found = []

    for skill in SKILLS:
        if skill in text:
            found.append(skill)

    return list(set(found))

# --------------------------
# EXPERIENCE DETECTION
# --------------------------
def estimate_experience(text):

    years = re.findall(
        r"(19\d{2}|20\d{2})",
        text
    )

    if len(years) < 2:
        return 0

    years = sorted(
        [int(y) for y in years]
    )

    return years[-1] - years[0]

# --------------------------
# SEMANTIC MATCHING
# --------------------------
def semantic_match(resume_text, jd_text):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [resume_text, jd_text]
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return round(similarity * 100, 2)

# --------------------------
# FORMAT SCORE
# --------------------------
def formatting_score(text):

    score = 100

    sections = [
        "experience",
        "education",
        "skills",
        "project"
    ]

    for section in sections:
        if section not in text.lower():
            score -= 10

    if len(text) < 500:
        score -= 20

    return max(score, 0)

# --------------------------
# ACHIEVEMENT SCORE
# --------------------------
def achievement_score(text):

    matches = re.findall(
        r"\d+%|\$\d+|\d+\+|\d+x",
        text
    )

    return min(len(matches) * 5, 100)

# --------------------------
# ATS SCORE
# --------------------------
def ats_score(
    semantic,
    skill_score,
    format_score,
    achievement
):

    score = (
        semantic * 0.40 +
        skill_score * 0.30 +
        format_score * 0.20 +
        achievement * 0.10
    )

    return round(score, 2)

# --------------------------
# FEEDBACK
# --------------------------
def generate_feedback(
    ats,
    missing_skills,
    exp
):

    tips = []

    if ats < 70:
        tips.append(
            "Improve ATS compatibility."
        )

    if missing_skills:
        tips.append(
            "Add missing skills: "
            + ", ".join(missing_skills)
        )

    if exp < 2:
        tips.append(
            "Highlight internships and projects."
        )

    if not tips:
        tips.append(
            "Strong resume."
        )

    return tips

# --------------------------
# UI
# --------------------------
st.title("🚀 Professional AI Resume Analyzer")

uploaded_resume = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=250
)

if st.button("Analyze Resume"):

    if uploaded_resume is None:
        st.error("Please upload a resume.")
        st.stop()

    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    resume_text = extract_pdf_text(
        uploaded_resume
    )

    resume_skills = extract_skills(
        resume_text
    )

    jd_skills = extract_skills(
        job_description
    )

    semantic = semantic_match(
        resume_text,
        job_description
    )

    matched = len(
        set(resume_skills)
        &
        set(jd_skills)
    )

    if len(jd_skills) > 0:
        skill_score = (
            matched /
            len(jd_skills)
        ) * 100
    else:
        skill_score = 0

    missing = list(
        set(jd_skills)
        -
        set(resume_skills)
    )

    fmt = formatting_score(
        resume_text
    )

    achievement = achievement_score(
        resume_text
    )

    experience = estimate_experience(
        resume_text
    )

    final_ats = ats_score(
        semantic,
        skill_score,
        fmt,
        achievement
    )

    feedback = generate_feedback(
        final_ats,
        missing,
        experience
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "ATS Score",
        f"{final_ats}%"
    )

    c2.metric(
        "Semantic Match",
        f"{semantic}%"
    )

    c3.metric(
        "Skill Match",
        f"{round(skill_score,2)}%"
    )

    c4.metric(
        "Experience",
        f"{experience} yrs"
    )

    st.subheader("Detected Skills")
    st.write(resume_skills)

    st.subheader("Missing Skills")
    st.write(missing)

    st.subheader("Recommendations")

    for item in feedback:
        st.write("✅", item)

    st.subheader("Resume Preview")

    st.text_area(
        "",
        resume_text[:5000],
        height=300
    )
```
