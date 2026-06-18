import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a credit card transaction dataset (CSV).")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    if "Class" not in data.columns:
        st.error("Dataset must contain a 'Class' column (0 = Genuine, 1 = Fraud).")
    else:
        X = data.drop("Class", axis=1)
        y = data["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.success(f"Model Accuracy: {accuracy:.4f}")

        st.subheader("Test Transaction Prediction")

        row_index = st.number_input(
            "Enter Test Row Number",
            min_value=0,
            max_value=len(X_test)-1,
            value=0
        )

        transaction = X_test.iloc[[row_index]]

        if st.button("Check Fraud"):
            prediction = model.predict(transaction)[0]

            if prediction == 1:
                st.error("🚨 Fraudulent Transaction Detected")
            else:
                st.success("✅ Genuine Transaction")
