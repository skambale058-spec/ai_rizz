import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Interview Coach", layout="wide")

st.title("🎤 AI Interview Prep Coach")

# ---------------- QUESTION BANK ----------------
QUESTIONS = {
    "Python Developer": [
        "What is Python and why is it used?",
        "Explain list vs tuple",
        "What is OOP in Python?",
        "What is decorators in Python?"
    ],
    "Data Science": [
        "What is machine learning?",
        "Difference between supervised and unsupervised learning?",
        "What is overfitting?",
        "Explain confusion matrix"
    ],
    "Web Developer": [
        "What is HTML?",
        "Difference between React and Angular?",
        "What is REST API?",
        "What is CSS?"
    ]
}

# ---------------- SCORE FUNCTION ----------------
def score_answer(question, answer):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([question, answer])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

# ---------------- UI ----------------
role = st.selectbox("Choose Role", list(QUESTIONS.keys()))

st.subheader("🧠 Answer the questions below")

answers = []
scores = []

for i, q in enumerate(QUESTIONS[role]):
    st.write(f"Q{i+1}. {q}")
    ans = st.text_area(f"Your Answer {i+1}", key=i)

    if ans:
        s = score_answer(q, ans)
        scores.append(s)
        answers.append((q, ans, s))

# ---------------- RESULT ----------------
if st.button("Submit Interview 🚀"):

    if not answers:
        st.warning("Please answer at least one question")
        st.stop()

    st.subheader("📊 Interview Results")

    avg_score = sum(scores) / len(scores)

    st.metric("Final Score", f"{round(avg_score,2)}%")

    for q, a, s in answers:
        st.write("------")
        st.write("❓", q)
        st.write("💬 Your Answer:", a)
        st.write("📊 Score:", s)

        if s < 50:
            st.error("Weak answer - add more explanation + keywords")
        elif s < 75:
            st.warning("Good but can be improved")
        else:
            st.success("Strong answer")

# ---------------- FEEDBACK ----------------
if scores:
    st.subheader("💡 Overall Feedback")

    if avg_score > 80:
        st.success("Excellent interview performance 👍")
    elif avg_score > 60:
        st.info("Good, but practice more technical depth")
    else:
        st.warning("Need strong improvement in answers")
