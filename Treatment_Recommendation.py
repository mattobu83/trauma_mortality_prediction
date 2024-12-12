import os
import streamlit as st
from openai import OpenAI

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

st.title("AI Treatment Option Advisor")
st.write("Enter patient details and symptoms to receive suggested treatment options.")

# Input fields
patient_age = st.number_input("Age", min_value=0, step=1)
patient_gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Other"])
symptoms = st.text_area("Symptoms")
medical_history = st.text_area("Relevant Medical History")

# Generate treatment suggestions
if st.button("Get Treatment Options"):
    if symptoms.strip():
        prompt = f"""
        A {patient_age}-year-old {patient_gender} patient presents with the following symptoms:
        {symptoms}.
        Relevant medical history includes: {medical_history}.
        What are the possible treatment options and their rationales?
        """
        try:
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.subheader("Suggested Treatment Options:")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter symptoms to proceed.")
