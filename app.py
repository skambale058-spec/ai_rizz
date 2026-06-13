import streamlit as st
import PyPDF2
import re
import numpy as np

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
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ==========================================
# INPUTS
# ==========================================

job_desc = st.text_area("Enter Job Description", height=200)
resume_input = st.text_area("Paste Resume Text (Optional)", height=200)

uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

# ==========================================
# SKILLS DATABASE (EXPANDABLE)
# ==========================================

skills_db = set([
    "python","java","sql","mysql","mongodb",
    "machine learning","deep learning","nlp",
    "data science","pandas","numpy","tensorflow",
    "pytorch","scikit-learn","flask","django",
    "fastapi","react","angular","javascript",
    "typescript","html","css","aws","azure",
    "gcp","docker","kubernetes","git","power bi",
    "tableau","excel"
])

# ==========================================
# PDF TEXT EXTRACTION
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
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9+.# ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==========================================
# SKILL EXTRACTION
# ==========================================

def extract_skills(text):
    found = set()

    for skill in skills_db:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found.add(skill)

    return sorted(found)


# ==========================================
# EMAIL EXTRACTION
# ==========================================

def extract_email(text):
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return emails[0] if emails else "Not Found"


# ==========================================
# PHONE EXTRACTION (IMPROVED)
# ==========================================

def extract_phone(text):
    phones = re.findall(r'(\+?\d[\d\s\-]{8,15}\d)', text)
    return phones[0] if phones else "Not Found"


# ==========================================
# ATS SEMANTIC SCORE
# ==========================================

def semantic_score(jd, resume):
    embeddings = model.encode([jd, resume])

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return max(0, min(score * 100, 100))


# ==========================================
# MAIN LOGIC
# ==========================================

if st.button("Analyze Resume"):

    if not job_desc.strip():
        st.error("Please enter a Job Description.")
        st.stop()

    # Resume input handling
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

    # CLEAN TEXT
    jd_clean = clean_text(job_desc)
    resume_clean = clean_text(resume_text)

    # CONTACT INFO
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)

    # SKILLS
    jd_skills = set(extract_skills(jd_clean))
    resume_skills = set(extract_skills(resume_clean))

    matched_skills = jd_skills & resume_skills
    missing_skills = jd_skills - resume_skills

    # SAFE SCORE CALCULATION
    skill_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0
    semantic = semantic_score(jd_clean, resume_clean)

    # FINAL SCORE (REALISTIC WEIGHTING)
    final_score = (0.65 * semantic) + (0.35 * skill_score)
    final_score = round(max(0, min(final_score, 100)), 2)

    # ==========================================
    # UI RESULTS
    # ==========================================

    st.header("📊 ATS Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Final ATS Score", f"{final_score}%")
    col2.metric("Semantic Match", f"{semantic:.2f}%")
    col3.metric("Skill Match", f"{skill_score:.2f}%")

    st.progress(int(final_score))

    # ==========================================
    # DETAILS
    # ==========================================

    st.subheader("📄 Resume Details")
    st.write("📧 Email:", email)
    st.write("📱 Phone:", phone)

    # ==========================================
    # SKILLS
    # ==========================================

    st.subheader("🎯 Required Skills")
    st.write(sorted(jd_skills))

    st.subheader("✅ Matched Skills")
    st.success(", ".join(matched_skills) if matched_skills else "None")

    st.subheader("❌ Missing Skills")
    st.error(", ".join(missing_skills) if missing_skills else "None")

    # ==========================================
    # RECOMMENDATIONS
    # ==========================================

    st.subheader("💡 Recommendations")

    if final_score >= 80:
        st.success("Excellent match for the role.")
    elif final_score >= 60:
        st.warning("Good match but can be improved.")
    else:
        st.error("Resume needs significant improvement.")

    if missing_skills:
        st.info("Add skills: " + ", ".join(missing_skills))

    if len(resume_text.split()) < 200:
        st.warning("Add more project/work experience details.")

    # ==========================================
    # INTERVIEW QUESTIONS
    # ==========================================

    st.subheader("🎤 Interview Questions")

    questions = []

    for skill in list(jd_skills)[:8]:
        questions.append(f"Explain your experience with {skill}.")
        questions.append(f"What real-world projects used {skill}?")

    if not questions:
        questions = [
            "Tell me about yourself",
            "Describe a challenging project",
            "Why this role?"
        ]

    for q in questions:
        st.write("•", q)

    # ==========================================
    # REPORT
    # ==========================================

    report = f"""
ATS RESUME REPORT

Final Score: {final_score}%
Semantic Score: {semantic:.2f}%
Skill Score: {skill_score:.2f}%

Email: {email}
Phone: {phone}

Matched Skills: {", ".join(matched_skills)}
Missing Skills: {", ".join(missing_skills)}
"""

    st.download_button(
        "📥 Download Report",
        report,
        file_name="ATS_Report.txt",
        mime="text/plain"
    )
