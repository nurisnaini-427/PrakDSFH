import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="NF1 Case Type Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 NF1 Case Type Prediction")
st.write("Masukkan data pasien untuk memprediksi jenis kasus NF1.")

# Input fitur
Axillary_Freckles = st.selectbox("Axillary Freckles", [0, 1])
Inguinal_Freckles = st.selectbox("Inguinal Freckles", [0, 1])
Lisch_Nodules = st.selectbox("Lisch Nodules", [0, 1])
Dermal_Neurofibromins = st.selectbox("Dermal Neurofibromins", [0, 1])
Plexiform_Neurofibromins = st.selectbox("Plexiform Neurofibromins", [0, 1])
Learning_Disability = st.selectbox("Learning Disability", [0, 1])
Astrocytoma = st.selectbox("Astrocytoma", [0, 1])
Hamartoma = st.selectbox("Hamartoma", [0, 1])
Scoliosis = st.selectbox("Scoliosis", [0, 1])
Other_Symptoms = st.selectbox("Other Symptoms", [0, 1])

if st.button("Predict"):

    data = pd.DataFrame([[
        Axillary_Freckles,
        Inguinal_Freckles,
        Lisch_Nodules,
        Dermal_Neurofibromins,
        Plexiform_Neurofibromins,
        Learning_Disability,
        Astrocytoma,
        Hamartoma,
        Scoliosis,
        Other_Symptoms
    ]], columns=[
        "Axillary_Freckles",
        "Inguinal_Freckles",
        "Lisch_Nodules",
        "Dermal_Neurofibromins",
        "Plexiform_Neurofibromins",
        "Learning_Disability",
        "Astrocytoma",
        "Hamartoma",
        "Scoliosis",
        "Other_Symptoms"
    ])

    # Standardisasi
    data_scaled = scaler.transform(data)

    # Prediksi
    prediction = model.predict(data_scaled)

    st.success(f"Hasil Prediksi Case Type: {prediction[0]}")
