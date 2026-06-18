import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Lost Child / Parent Finder")

st.title("🔍 Lost Child / Parent Finder")

DATA_FILE = "reports.csv"

if not os.path.exists(DATA_FILE):
    pd.DataFrame(
        columns=["Type", "Name", "Age", "City", "Description", "Contact"]
    ).to_csv(DATA_FILE, index=False)

menu = st.sidebar.selectbox(
    "Select Option",
    ["Report Missing Person", "Search Person"]
)

if menu == "Report Missing Person":

    st.header("Submit Report")

    person_type = st.selectbox(
        "Type",
        ["Lost Child", "Lost Parent", "Found Child", "Found Parent"]
    )

    name = st.text_input("Name")
    age = st.number_input("Age", 0, 120)
    city = st.text_input("City")
    description = st.text_area("Description")
    contact = st.text_input("Contact Number")

    if st.button("Save Report"):
        df = pd.read_csv(DATA_FILE)

        new_row = pd.DataFrame([{
            "Type": person_type,
            "Name": name,
            "Age": age,
            "City": city,
            "Description": description,
            "Contact": contact
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)

        st.success("Report Saved Successfully!")

else:

    st.header("Search Person")

    search_city = st.text_input("Enter City")
    search_age = st.number_input("Enter Age", 0, 120)

    if st.button("Search"):
        df = pd.read_csv(DATA_FILE)

        results = df[
            (df["City"].str.lower() == search_city.lower()) &
            (abs(df["Age"] - search_age) <= 5)
        ]

        if len(results) > 0:
            st.success(f"{len(results)} Match Found")
            st.dataframe(results)
        else:
            st.warning("No Match Found")
