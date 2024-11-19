import altair as alt
import pandas as pd
import streamlit as st


# Show the page title and description.
st.set_page_config(page_title="Trauma Mortality Prediction Calculator", page_icon="🏥", initial_sidebar_state='collapsed')
st.title("🏥 Trauma Mortality Prediction Calculator")
st.write(
    """
    This app visualizes trauma hospital stay data from Trinetx.
    Mortality prediction can be done using selected patient attributes.
    """
)

st.markdown("# Patient Demographics")

# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    df = pd.read_csv("data/530_project_race_group_stats.csv")
    return df


df = load_data()
df = df.rename(columns={"NonBinary": "NonBinaryCount"})
# Show a multiselect widget with the genres using `st.multiselect`.
races = st.multiselect(
    "Races",
    df.CombinedRace.unique(),
    ['AMERICANINDIAN', 'ASIAN', 'BLACK', 'PACIFICISLANDER', 'RACEOTHER', 'WHITE', 'RACE_UK'],
)

# Show a slider widget with the years using `st.slider`.
#years = st.slider("Years", 1986, 2006, (2000, 2016))

# Filter the dataframe based on the widget input and reshape it.
df_filtered = df[(df["CombinedRace"].isin(races))]
df_reshaped = df_filtered


# Display the data as a table using `st.dataframe`.
st.dataframe(
    df_reshaped,
    use_container_width=True
)

# Display the data as an Altair chart using `st.altair_chart`.
# Convert to long format for Altair
df_long = df_reshaped.melt(id_vars='CombinedRace', value_vars=['MaleCount', 'FemaleCount','NonBinaryCount'], 
                  var_name='Gender', value_name='Count')
#print(df_long)
# Create the stacked bar chart
chart = alt.Chart(df_long).mark_bar().encode(
    y=alt.Y('CombinedRace:N', title='Race'),
    x=alt.X('Count:Q', stack='normalize', title='Proportion of Patients'),
    color=alt.Color('Gender:N', title='Gender'),
    tooltip=['CombinedRace', 'Gender', 'Count']
).properties(
    width=600,
    height=300,
    title='Proportion of Gender per Race'
)


df_reshaped['DeathProportion'] = df_reshaped['DeathCount'] / df_reshaped['TotalPatients']

# Create the death proportion chart
death_chart = alt.Chart(df_reshaped).mark_bar().encode(
    y=alt.Y('CombinedRace:N', title='Race'),
    x=alt.X('DeathProportion:Q', title='Proportion of Deaths', axis=alt.Axis(format='%')),
    color=alt.value('red'),  # Color the death chart bars red for clarity
    tooltip=['CombinedRace', 'DeathCount', 'DeathProportion']
).properties(
    width=600,
    height=400,
    title='Proportion of Patient Deaths per Race'
)

option = st.selectbox('Select a Data Visualization',["Gender Distribution", "Death Proportion"])
if option == "Gender Distribution":
    st.altair_chart(chart, use_container_width=True)
else:
    st.altair_chart(death_chart, use_container_width=True)