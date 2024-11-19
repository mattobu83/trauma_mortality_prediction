import streamlit as st



st.markdown("# 💀 Calculator")
st.sidebar.markdown("# 💀 Calculator")
st.write(
    """
    To predict a patient's probability of death, enter in the following variables
    """
)
age = st.slider('Age')
race = st.selectbox("Races",['AMERICANINDIAN', 'ASIAN', 'BLACK', 'PACIFICISLANDER', 'RACEOTHER', 'WHITE', 'RACE_UK'] )
trauma_type = st.selectbox("Trauma Type",['Car Accident', 'Gun', 'Fall'] )
transport = st.selectbox("Transportation Method", ["Ambulance","Medivac", "Self"])
como = st.selectbox("Comorbidites",["Heart Disease"])
hosp_event = st.slider("Number of Hospital Events")

#add function here to predict using the saved model weights