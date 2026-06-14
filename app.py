import streamlit as st

st.set_page_config(
    page_title="AI Career Roadmap Generator",
    layout="wide"
)

st.title("🚀 AI Career Roadmap Generator")

# ---------------- CAREER DATA ----------------
CAREER_MAP = {
    "Web Developer": {
        "Beginner": ["HTML", "CSS", "JavaScript Basics"],
        "Intermediate": ["React", "APIs", "Git"],
        "Advanced": ["System Design", "Performance Optimization"]
    },
    "Data Scientist": {
        "Beginner": ["Python", "Math Basics", "Pandas"],
        "Intermediate": ["Machine Learning", "SQL", "EDA"],
        "Advanced": ["Deep Learning", "NLP", "Model Deployment"]
    },
    "Cyber Security": {
        "Beginner": ["Networking Basics", "Linux", "Security Fundamentals"],
        "Intermediate": ["Ethical Hacking", "Firewalls", "Security Tools"],
        "Advanced": ["Pen Testing", "Threat Analysis"]
    }
}

# ---------------- INPUT ----------------
field = st.selectbox("Select Career Field", list(CAREER_MAP.keys()))
level = st.selectbox("Select Level", ["Beginner", "Intermediate", "Advanced"])

skills_input = st.text_area("Enter your current skills (comma separated)")

# ---------------- LOGIC ----------------
def generate_roadmap(field, level, skills_input):

    required_skills = CAREER_MAP[field][level]

    user_skills = [
        s.strip().lower()
        for s in skills_input.split(",")
        if s.strip() != ""
    ]

    missing_skills = [
        skill for skill in required_skills
        if skill.lower() not in user_skills
    ]

    return required_skills, missing_skills

# ---------------- BUTTON ----------------
if st.button("Generate Roadmap 🚀"):

    required, missing = generate_roadmap(
        field,
        level,
        skills_input
    )

    st.subheader("📚 Learning Roadmap")

    for i, skill in enumerate(required):
        st.write(f"Step {i+1}: {skill}")

    st.subheader("❌ Missing Skills")

    if missing:
        for m in missing:
            st.error(m)
    else:
        st.success("You already know all required skills!")

    # ---------------- PROGRESS ----------------
    progress = 100 - (len(missing) / len(required) * 100)

    st.subheader("📊 Progress")

    st.progress(int(progress))

    st.metric("Completion", f"{int(progress)}%")
