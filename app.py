import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Job Copilot", layout="wide")

st.title("🚀 AI Job Hunting Copilot (Pro Max)")

# ---------------- JOB ROLES ----------------
JOBS = {
    "Backend Developer": "python django fastapi databases api docker sql",
    "Frontend Developer": "html css javascript react ui ux frontend",
    "Data Scientist": "python machine learning deep learning pandas numpy sql",
    "DevOps Engineer": "aws docker kubernetes ci cd linux cloud",
    "Cyber Security Analyst": "network security ethical hacking firewall linux"
}

# ---------------- MATCH ENGINE ----------------
def match_score(resume, job_desc):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume, job_desc])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

# ---------------- RESUME INPUT ----------------
resume = st.text_area("Paste Your Resume Text")

if st.button("Analyze Career Fit 🚀"):

    if not resume.strip():
        st.warning("Please enter resume text")
        st.stop()

    results = []

    for role, desc in JOBS.items():
        score = match_score(resume.lower(), desc)
        results.append((role, score))

    results.sort(key=lambda x: x[1], reverse=True)

    st.subheader("📊 Your Best Career Matches")

    for role, score in results:
        if score > 70:
            st.success(f"{role} → {score}% Fit")
        elif score > 40:
            st.info(f"{role} → {score}% Fit")
        else:
            st.error(f"{role} → {score}% Fit")

    best_role = results[0][0]

    st.subheader("🎯 Best Recommended Role")
    st.success(best_role)

    # ---------------- INTERVIEW QUESTIONS ----------------
    st.subheader("🎤 Interview Questions")

    questions = {
        "Backend Developer": [
            "Explain REST API",
            "What is database indexing?",
            "What is microservices?"
        ],
        "Frontend Developer": [
            "Difference between React and Angular",
            "What is DOM?",
            "Explain state vs props"
        ],
        "Data Scientist": [
            "What is overfitting?",
            "Explain ML pipeline",
            "Difference between AI and ML"
        ],
        "DevOps Engineer": [
            "What is Docker?",
            "Explain CI/CD",
            "What is Kubernetes?"
        ],
        "Cyber Security Analyst": [
            "What is phishing?",
            "Explain firewall",
            "What is encryption?"
        ]
    }

    for q in questions.get(best_role, []):
        st.write("•", q)

    # ---------------- CAREER ADVICE ----------------
    st.subheader("💡 Career Advice")

    if results[0][1] > 80:
        st.success("You are highly job ready 🚀")
    elif results[0][1] > 50:
        st.info("Improve a few skills to become job ready")
    else:
        st.warning("Focus on learning core skills first")
