import streamlit as st

st.set_page_config(page_title="AI Career Roadmap", layout="wide")

st.title("🚀 AI Career Roadmap Generator")

# ---------------- SKILL MAP ----------------
CAREER_MAP = {
    "Web Developer": {
        "Beginner": ["HTML", "CSS", "JavaScript basics"],
        "Intermediate": ["React", "APIs", "Git"],
        "Advanced": ["System Design", "Performance Optimization"]
    },
    "Data Scientist": {
        "Beginner": ["Python", "Math Basics", "Pandas"],
        "Intermediate": ["Machine Learning", "SQL", "EDA"],
        "Advanced": ["Deep Learning", "NLP", "Model Deployment"]
    },
    "Cyber Security": {
        "Beginner": ["Networking basics", "Linux", "Security basics"],
        "Intermediate": ["Ethical Hacking", "Firewalls", "Tools"],
        "Advanced": ["Pen Testing", "Threat Analysis"]
    }
}

# ---------------- UI ----------------
field = st.selectbox("Select Career Field", list(CAREER_MAP.keys()))
level = st.selectbox("Select Level", ["Beginner", "Intermediate", "Advanced"])

current_skills = st.text_area("Enter your current skills (comma separated)")

# ---------------- ROADMAP ----------------
def generate_roadmap(field, level, skills):
    required = CAREER_MAP[field][level]

    user_skills = [s.strip().lower() for s in skills.split(",") if s]

    missing = [r for r in required if r.lower() not in user_skills]

    return required, missing

# ---------------- OUTPUT ----------------
if st.button("Generate Roadmap 🚀"):

    required, missing = generate_roadmap(field, level, current_skills)

    st.subheader("📚 Learning Roadmap")

    for i, r in enumerate(required):
        st.write(f"Step {i+1}: {r}")

    st.subheader("❌ Missing Skills")

    if missing:
        for m in missing:
            st.error(m)
    else:
        st.success("You already know all required skills!")

    progress = 100 - (len(missing) / len(required) * 100)

    st.subheader("📊 Progress")
    st.progress(int(progress))

    st.metric("Completion", f"{int(progress)}%")n answers")
