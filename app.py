import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1. Embedded Dataset
# -----------------------------
data = {
    "text": [
        "Government launches new education policy",
        "Aliens landed in India last night confirmed",
        "WHO approves new vaccine for public use",
        "Drinking petrol cures diseases viral claim",
        "Stock market reaches new high today",
        "Celebrities using blood therapy for youth",
        "India wins cricket world cup final",
        "Earth is flat according to viral video",
        "Scientists discover new planet similar to Earth",
        "Chocolate can cure cancer instantly"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Train Model (inside app)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake News Detector")

st.title("📰 Fake News Detection App (Single File)")
st.write("Enter any news text and check if it is REAL or FAKE")

news = st.text_area("Enter News Here")

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter some text")
    else:
        vec = vectorizer.transform([news])
        result = model.predict(vec)[0]

        if result == 1:
            st.success("REAL NEWS ✅")
        else:
            st.error("FAKE NEWS ❌")
