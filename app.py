import streamlit as st
import PyPDF2
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# PAGE CONFIG
# --------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screening System")

# --------------------------
# INPUTS
# --------------------------

job_desc = st.text_area(
    "Enter Job Description",
    height=200
)

resume_input = st.text_area(
    "Paste Resume Text (Optional)",
    height=200
)

uploaded_file = st.file_uploader(
    "Upload Resume PDF (Optional)",
    type=["pdf"]
)

# --------------------------
# SKILLS DATABASE
# --------------------------

skills_db = [
    "python",
    "java",
    "sql",
    "mysql",
    "mongodb",
    "machine learning",
    "deep learning",
    "nlp",
    "artificial intelligence",
    "data science",
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "flask",
    "django",
    "fastapi",
    "react",
    "angular",
    "javascript",
    "typescript",
    "html",
    "css",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "github",
    "power bi",
    "tableau",
    "excel"
]

# --------------------------
# PDF READER
# --------------------------

def extract_text_from_pdf(pdf_file):
    text = ""

    try:
        reader = PyPDF2.PdfReader(pdf_file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    except Exception:
        return ""

    return text


# --------------------------
# CLEAN TEXT
# --------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# --------------------------
# SKILL EXTRACTION
# --------------------------

def extract_skills(text):
    text = text.lower()

    found = []

    for skill in skills_db:
        if skill.lower() in text:
            found.append(skill)

    return sorted(list(set(found)))


# --------------------------
# EMAIL / PHONE
# --------------------------

def extract_email(text):
    emails = re.findall(
        r'[\w\.-]+@[\w\.-]+\.\w+',
        text
    )
    return emails[0] if emails else "Not Found"


def extract_phone(text):
    phones = re.findall(
        r'\+?\d[\d\s\-]{8,15}',
        text
    )
    return phones[0] if phones else "Not Found"


# --------------------------
# ANALYZE
# --------------------------

if st.button("Analyze Resume"):

    if not job_desc.strip():
        st.error("Please enter a Job Description.")
        st.stop()

    resume_text = ""

    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)

    elif resume_input.strip():
        resume_text = resume_input

    else:
        st.error(
            "Please upload a PDF or paste resume text."
        )
        st.stop()

    if len(resume_text.strip()) == 0:
        st.error(
            "Unable to extract text from resume."
        )
        st.stop()

    # --------------------------
    # CLEAN
    # --------------------------

    clean_jd = clean_text(job_desc)
    clean_resume = clean_text(resume_text)

    # --------------------------
    # PARSE RESUME
    # --------------------------

    email = extract_email(resume_text)
    phone = extract_phone(resume_text)

    # --------------------------
    # SKILLS
    # --------------------------

    resume_skills = extract_skills(clean_resume)
    jd_skills = extract_skills(clean_jd)

    matched_skills = sorted(
        list(set(resume_skills) & set(jd_skills))
    )

    missing_skills = sorted(
        list(set(jd_skills) - set(resume_skills))
    )

    if len(jd_skills) > 0:
        skill_score = (
            len(matched_skills) /
            len(jd_skills)
        ) * 100
    else:
        skill_score = 0

    # --------------------------
    # SEMANTIC MATCHING
    # --------------------------

    with st.spinner("Calculating ATS score..."):

        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        embeddings = model.encode(
            [clean_jd, clean_resume]
        )

        semantic_score = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0] * 100

    # --------------------------
    # FINAL SCORE
    # --------------------------

    final_score = (
        semantic_score * 0.70 +
        skill_score * 0.30
    )

    final_score = min(
        max(final_score, 0),
        100
    )

    # --------------------------
    # RESULTS
    # --------------------------

    st.header("📊 ATS Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Final ATS Score",
        f"{final_score:.2f}%"
    )

    col2.metric(
        "Semantic Match",
        f"{semantic_score:.2f}%"
    )

    col3.metric(
        "Skill Match",
        f"{skill_score:.2f}%"
    )

    # --------------------------
    # RESUME DETAILS
    # --------------------------

    st.subheader("📄 Resume Details")

    st.write("Email:", email)
    st.write("Phone:", phone)

    # --------------------------
    # JD ANALYSIS
    # --------------------------

    st.subheader("🎯 Job Description Analysis")

    st.write("Required Skills:")
    st.write(jd_skills)

    # --------------------------
    # SKILL MATCHING
    # --------------------------

    st.subheader("✅ Matched Skills")

    if matched_skills:
        st.success(", ".join(matched_skills))
    else:
        st.warning("No matching skills found.")

    st.subheader("❌ Missing Skills")

    if missing_skills:
        st.error(", ".join(missing_skills))
    else:
        st.success("No missing skills detected.")

    # --------------------------
    # RECOMMENDATIONS
    # --------------------------

    recommendations = []

    if final_score < 50:
        recommendations.append(
            "Resume needs significant improvement."
        )

    elif final_score < 70:
        recommendations.append(
            "Resume partially matches the role."
        )

    else:
        recommendations.append(
            "Resume is a strong match."
        )

    if missing_skills:
        recommendations.append(
            "Add these skills if applicable: "
            + ", ".join(missing_skills)
        )

    if len(resume_text.split()) < 250:
        recommendations.append(
            "Resume appears short. Add more project and experience details."
        )

    st.subheader("💡 Recommendations")

    for item in recommendations:
        st.write("•", item)

    # --------------------------
    # INTERVIEW QUESTIONS
    # --------------------------

    st.subheader("🎤 Suggested Interview Questions")

    questions = []

    for skill in jd_skills[:10]:
        questions.append(
            f"Describe your experience with {skill}."
        )
        questions.append(
            f"What challenges have you faced while using {skill}?"
        )

    if not questions:
        questions = [
            "Tell us about yourself.",
            "Describe a challenging project.",
            "Why are you interested in this role?"
        ]

    for q in questions:
        st.write("•", q)

    # --------------------------
    # RESUME PREVIEW
    # --------------------------

    st.subheader("📑 Resume Preview")

    st.text_area(
        "",
        resume_text[:5000],
        height=250
    )

    # --------------------------
    # EXPORT REPORT
    # --------------------------

    report = f"""
ATS RESUME ANALYSIS REPORT

Final ATS Score: {final_score:.2f}%
Semantic Match: {semantic_score:.2f}%
Skill Match: {skill_score:.2f}%

Email: {email}
Phone: {phone}

Required Skills:
{", ".join(jd_skills)}

Matched Skills:
{", ".join(matched_skills)}

Missing Skills:
{", ".join(missing_skills)}

Recommendations:
{chr(10).join(recommendations)}
"""

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="ATS_Report.txt",
        mime="text/plain"
    )
