import streamlit as st
import PyPDF2
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screening System")
st.markdown("Upload a resume and compare it with a Job Description using AI.")

# ==========================================
# LOAD MODEL (CACHE)
# ==========================================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

model = load_model()

# ==========================================
# INPUTS
# ==========================================

job_desc = st.text_area(
    "Enter Job Description",
    height=200
)

resume_input = st.text_area(
    "Paste Resume Text (Optional)",
    height=200
)

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

# ==========================================
# SKILLS DATABASE
# ==========================================

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

# ==========================================
# PDF READER
# ==========================================

def extract_text_from_pdf(pdf_file):
    text = ""

    try:
        reader = PyPDF2.PdfReader(pdf_file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:
        st.error(f"PDF Error: {e}")
        return ""

    return text


# ==========================================
# CLEAN TEXT
# ==========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==========================================
# SKILL EXTRACTION
# ==========================================

def extract_skills(text):

    found = []

    for skill in skills_db:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


# ==========================================
# EMAIL EXTRACTION
# ==========================================

def extract_email(text):

    emails = re.findall(
        r'[\w\.-]+@[\w\.-]+\.\w+',
        text
    )

    return emails[0] if emails else "Not Found"


# ==========================================
# PHONE EXTRACTION
# ==========================================

def extract_phone(text):

    phones = re.findall(
        r'(\+?\d[\d\s\-]{8,15}\d)',
        text
    )

    return phones[0] if phones else "Not Found"


# ==========================================
# ANALYZE BUTTON
# ==========================================

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
        st.error("Upload a PDF or paste resume text.")
        st.stop()

    if not resume_text.strip():
        st.error("Unable to extract resume text.")
        st.stop()

    # ==========================================
    # CLEAN
    # ==========================================

    clean_jd = clean_text(job_desc)
    clean_resume = clean_text(resume_text)

    # ==========================================
    # CONTACT INFO
    # ==========================================

    email = extract_email(resume_text)
    phone = extract_phone(resume_text)

    # ==========================================
    # SKILLS
    # ==========================================

    jd_skills = extract_skills(clean_jd)
    resume_skills = extract_skills(clean_resume)

    matched_skills = sorted(
        set(jd_skills) & set(resume_skills)
    )

    missing_skills = sorted(
        set(jd_skills) - set(resume_skills)
    )

    if len(jd_skills) > 0:
        skill_score = (
            len(matched_skills)
            / len(jd_skills)
        ) * 100
    else:
        skill_score = 0

    # ==========================================
    # SEMANTIC SCORE
    # ==========================================

    with st.spinner("Calculating ATS Score..."):

        embeddings = model.encode(
            [clean_jd, clean_resume]
        )

        semantic_score = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0] * 100

    # ==========================================
    # FINAL SCORE
    # ==========================================

    final_score = (
        semantic_score * 0.70
        + skill_score * 0.30
    )

    final_score = max(
        min(final_score, 100),
        0
    )

    # ==========================================
    # RESULTS
    # ==========================================

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

    st.progress(int(final_score))

    # ==========================================
    # CONTACT INFO
    # ==========================================

    st.subheader("📄 Resume Details")

    st.write("📧 Email:", email)
    st.write("📱 Phone:", phone)

    # ==========================================
    # JD SKILLS
    # ==========================================

    st.subheader("🎯 Required Skills")

    st.write(jd_skills)

    # ==========================================
    # MATCHED SKILLS
    # ==========================================

    st.subheader("✅ Matched Skills")

    if matched_skills:
        st.success(", ".join(matched_skills))
    else:
        st.warning("No matching skills found.")

    # ==========================================
    # MISSING SKILLS
    # ==========================================

    st.subheader("❌ Missing Skills")

    if missing_skills:
        st.error(", ".join(missing_skills))
    else:
        st.success("No missing skills detected.")

    # ==========================================
    # RECOMMENDATIONS
    # ==========================================

    st.subheader("💡 Recommendations")

    recommendations = []

    if final_score >= 80:
        recommendations.append(
            "Excellent match for the role."
        )

    elif final_score >= 60:
        recommendations.append(
            "Good match but can be improved."
        )

    else:
        recommendations.append(
            "Resume requires significant improvement."
        )

    if missing_skills:
        recommendations.append(
            "Add relevant missing skills: "
            + ", ".join(missing_skills)
        )

    if len(resume_text.split()) < 250:
        recommendations.append(
            "Add more project and work experience details."
        )

    for item in recommendations:
        st.write("•", item)

    # ==========================================
    # INTERVIEW QUESTIONS
    # ==========================================

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

    # ==========================================
    # RESUME PREVIEW
    # ==========================================

    st.subheader("📑 Resume Preview")

    st.text_area(
        "",
        resume_text[:5000],
        height=250
    )

    # ==========================================
    # REPORT
    # ==========================================

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
        "📥 Download Report",
        report,
        file_name="ATS_Report.txt",
        mime="text/plain"
    )
   
