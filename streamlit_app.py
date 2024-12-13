import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from openai import OpenAI
from prediction_maker import *
import pickle
import xgboost as xgb

model_path = "xgboost_patient_death_predictor_percentage_reduced.pkl"
# Set up the Streamlit page with favicon
st.set_page_config(
    page_title="Trauma Prediction App",
    page_icon="1stonybrook.jpg",  # Replace with your icon file
    layout="wide"
)

# Title and Description
st.title("🏥 SAVIOR: Beyond-Human Trauma Insight")
st.write(
    """
    This advanced tool not only predicts trauma mortality risk but also provides 
    extraordinarily sophisticated intervention recommendations that push beyond conventional human reasoning. 
    Utilizing cutting-edge large language models and integrated medical scoring, 
    **SAVIOR** aims to guide life-saving decisions.
    """
)

# Load GPT model for recommendations
token = os.environ.get("GITHUB_TOKEN", "")
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

# Layout: Top and Bottom Columns
top_col1, top_col2 = st.columns([1, 1])
bottom_col1, bottom_col2 = st.columns([1, 1])

# Top Left Column: Input Form
with top_col1:
    st.markdown("## 💀 Patient Details")
    # Core Demographics
    age_input = st.slider("Patient Age", 0, 120, 50)
    race_input = st.selectbox("Patient Race", options=["White", "Black", "Hispanic", "Asian", "Other"])
    gender_input = st.selectbox("Patient Gender", ["Male", "Female", "Non-Binary"])
    transport_method = st.selectbox("Transport Method", ["Ambulance", "Heliambulance", "Self Car"])
    trauma_type = st.selectbox("Trauma Type",['Blunt', 'Burn', 'Penetrating', 
                                        'Storm Related','neglect or abuse', 'terrorism'])
    iss = st.slider("Injury Severity", 0, 150, 50)
    heart_attack = st.checkbox("Heart Attack Prior to arrival?")
    suppo2 = st.checkbox("Patient Given Supplementary O2")
    # Additional Vital Parameters
    gcs_motor = st.slider("Glasgow Coma Scale (GCS) - Motor", 1, 6, 6)
    gcs_verbal = st.slider("Glasgow Coma Scale (GCS) - Verbal", 1, 5, 5)  
    gcs_eye = st.slider("Glasgow Coma Scale (GCS) - Eye", 1, 4, 4) 
    resp_rate = st.slider("Respiratory Rate (breaths/min)", 0, 60, 20)
    blood_pressure_systolic = st.slider("Systolic Blood Pressure (mmHg)", 50, 250, 120)
    heart_rate = st.slider("Heart Rate (bpm)", 0, 200, 80)
    
    symptoms = st.text_area("Symptoms")
    medical_history = st.text_area("Relevant Medical History")

# Top Right Column: Prediction Result and Pie Chart
with top_col2:
    # Convert categorical inputs to numeric codes (simulate model input)
    race_mapping = {race: idx for idx, race in enumerate(["White", "Black", "Hispanic", "Asian", "Other"])}
    gender_mapping = {"Male": 1, "Female": 2, "Non-Binary": 3}
    transport_mapping = {"Ambulance": 0, "Heliambulance": 1, "Self Car": 2}
    trauma_mapping = {injury: index for index, injury in enumerate(['Blunt', 'Burn', 'Penetrating', 
                                        'Storm Related','neglect or abuse', 'terrorism'])}
    heart_attack_mapping = 2 if ~heart_attack else 1
    suppo2_mapping = 2 if suppo2 else 1
    model, feature_columns = load_model_and_features(model_path)
    
    # Example user input based on the feature columns
    user_input = {'GCSMOTOR': gcs_motor,
                'GCSVERBAL': gcs_verbal,
                'GCSEYE':gcs_eye,
                'PREHOSPITALCARDIACARREST': heart_attack_mapping,
                'AgeYears':age_input,
                'SEX':gender_mapping[gender_input],
                'TRAUMATYPE': trauma_mapping[trauma_type],
                'SUPPLEMENTALOXYGEN':suppo2_mapping,
                'VTEPROPHYLAXISTYPE':11,
                "ISS":iss,
                "MECHANISM":1, 
                "SBP":blood_pressure_systolic,
                "PULSEOXIMETRY":100 }
    # Run prediction
    prediction_proba = predict_percentage(user_input, model, feature_columns)
    
    # Advanced placeholder: a pseudo "trauma scoring" that reduces the random variance by factoring in vitals
    # For demonstration, let's say we weigh GCS and blood pressure for mortality risk:
    #base_risk = random.uniform(0.1, 0.9)
    # Adjust the base risk considering vital signs (lower GCS or very low BP → higher risk)
    #gcs_factor = (15 - gcs_input) / 15  # scaled from 0 (perfect) to ~0.8 (worst)
    #bp_factor = 1 if blood_pressure_systolic < 90 else 0.5 if blood_pressure_systolic < 120 else 0.3
    #final_risk = (base_risk + gcs_factor * 0.4 + bp_factor * 0.2) / 1.6  # Normalized


    #prediction_proba = min(max(final_risk, 0), 1)  # Ensure within [0,1]
    mortality_risk = (prediction_proba) * 100
    # Display prediction result
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 40px; color: red; font-weight: bold;">
            Mortality Risk: {mortality_risk:.2f}%
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pie chart visualization
    st.markdown("### Mortality Risk Breakdown")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [prediction_proba, 1 - prediction_proba],
        labels=["Mortality Risk", "Survival Probability"],
        autopct="%1.1f%%",
        colors=["#FF5733", "#33FF57"],
        startangle=90
    )
    st.pyplot(fig)

# Bottom Left Column: Patient Demographics and Vitals Summary
with bottom_col1:
    st.markdown("### Comprehensive Patient Summary")
    st.write(
        f"**Age**: {age_input}\n\n"
        f"**Race**: {race_input}\n\n"
        f"**Gender**: {gender_input}\n\n"
        f"**Transport Method**: {transport_method}\n\n"
        f"**GCS - Motor**: {gcs_motor}\n\n"
        f"**GCS- Verbal**: {gcs_verbal}\n\n"
        f"**GCS - Eye**: {gcs_eye}\n\n"
        f"**Respiratory Rate**: {resp_rate} breaths/min\n\n"
        f"**Systolic Blood Pressure**: {blood_pressure_systolic} mmHg\n\n"
        f"**Heart Rate**: {heart_rate} bpm\n\n"
        f"**Symptoms**: {symptoms}\n\n"
        f"**Relevant Medical History**: {medical_history}"
    )

# Bottom Right Column: Generated Treatment Options
with bottom_col2:
    if st.button("SAVE A LIFE"):
        if symptoms.strip():
            # Enhanced prompt: structured instructions for the model to think more comprehensively
            prompt = f"""
            Analyze the following patient presentation deeply and beyond standard reasoning:
            Patient Details:
            - Age: {age_input}-year-old
            - Gender: {gender_input}
            - Race: {race_input}
            - Transport Method: {transport_method}
            - GCS - Motor: {gcs_motor}
            - GCS - Eye: {gcs_eye}
            - GCS - Verbal: {gcs_verbal}
            - Respiratory Rate: {resp_rate}
            - Systolic Blood Pressure: {blood_pressure_systolic}
            - Heart Rate: {heart_rate}
            
            Symptoms: {symptoms}.
            Medical History: {medical_history}.

            Instructions:
            1. Identify potential life-threatening conditions, including rare or obscure diagnoses that human clinicians might overlook.
            2. Specify immediate life-saving interventions that exceed routine protocols—consider experimental or advanced treatments if standard measures fail.
            3. Suggest a comprehensive set of diagnostic tests, imaging, labs, or emerging biomarkers to confirm suspected conditions, going beyond typical ER protocols.
            4. Recommend resource allocation (specialist teams, advanced ICU equipment, telemedicine consultation with global experts) anticipating complex complications.
            5. The response should maintain a purely clinical and highly analytical tone—no extraneous text, no disclaimers. Present the reasoning as if you are a top-tier trauma surgeon supported by an advanced AI reasoning engine.

            Remember: Provide your reasoning and recommendations at a level that transcends standard human clinical practice, integrating cutting-edge medical insights.
            """

            with st.spinner("The AI is formulating a beyond-human response... TO SAVE A LIFE"):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a highly advanced medical AI integrated with state-of-the-art trauma knowledge."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter symptoms to proceed.")
    else:
        st.info("Adjust patient details and then click 'SAVE A LIFE' for recommendations.")

# Footer and Disclaimers
st.markdown("---")
st.write(
    "Built by Stony Brook folks (Mittal, Shivkar, Matt) | [GitHub Repository](https://github.com/your-repo)\n\n"
    "_Note: This tool is for demonstration and educational purposes, pushing the boundaries of AI-driven medical reasoning. Always consult real-world medical professionals._"
)
