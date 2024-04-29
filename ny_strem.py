
import pandas as pd
import json
import streamlit as st
import openai
from openai import OpenAI

API_KEY = open('Open_AI_key', 'r').read()

client = OpenAI(
    api_key=API_KEY
) 

# Läs in API-nyckeln från filen
with open("Open_AI_key", "r") as file:
    api_key = file.read().strip()

# Ange din API-nyckel
openai.api_key = api_key

# Load the JSON file into a DataFrame 
lines = []
with open('dataset.jsonl', 'rb') as file: 
    for i, line in enumerate(file):
        lines.append(line.strip())
        if i >= 9999:
            break

# Convert each line from JSON format to Python dictionary
data = [json.loads(line) for line in lines]

# If the JSON file has nested structures, pandas will automatically flatten them
jobtech_dataset = pd.json_normalize(data)

#select only these columns
subset = jobtech_dataset[[
    'id',
    'original_id',
    'headline',
    'number_of_vacancies',
    'experience_required',
    'driving_license_required',
    'detected_language',
    'description.text',
    'duration.label',
    'working_hours_type.label',
    'employer.name',
    'employer.workplace',
    'workplace_address.municipality',
    'workplace_address.region',
    'workplace_address.region_code',
    'keywords.extracted.occupation'
]]

#Titel och text högst upp
st.markdown("<h1 style='color: red; display: inline;'>ATH</h1><h1 style='color: black; display: inline;'>WORK</h1>", unsafe_allow_html=True)

st.markdown("Info om vårt projekt")
st.markdown("---")
st.write(subset)



#Den gråa sidopanelen
vidare_lasning = """Text om vi vill ha...

[Swedish Elite Sport](https://www.idan.dk/media/stgjthhj/swedish-elite-sport.pdf) handlar om ...

[How 5 Athletes Afford to Stay in the Game and Still Make Rent](https://www.thecut.com/2024/01/pro-athletes-working-second-jobs-careers.html) handlar om..."""

kontakt_uppgifter = """
Python Consulant Vera Hertzman
Vera@outlook.com
+46 0000000

Head of AI Thea Håkansson
Thea@gmail.se
+46 00000000

Head of Streamlit Frida Eriksson
Royal@yahoo.com
+46 0000000

Project Coordinator Miranda Tham
Miranda@hotmail.com
+46 0000000

Agenda and Report Administrator Tove Lennertsson
Tove@gmail.com
+46 0000000"""

left_column = st.sidebar

left_column.markdown("""
<style>
.left-column {
    background-color: #f0f0f0;
    width: 30%;
    padding: 20px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

#Texten i sidopanelen: annan text som vi kan lägga till
left_column.markdown("### Fri text")
left_column.markdown("Text...")

#Vidare läsning i sidopanelen
with left_column.expander("Vidare läsning"):
    st.write(vidare_lasning)

with left_column.expander("Kontaktuppgifter"):
    st.write(kontakt_uppgifter)


# Display description of a specific row
row_index = st.slider("Select Row Index", 0, len(subset)-1, 25)
st.subheader("Description for Selected Row:")
st.write(subset['description.text'].iloc[row_index])

# Show the variables in the dataset (equivalent to column names)
st.write("Columns in the dataset:")
st.write(subset.columns)
st.write('')


#Tabell där man kan filtrera med båda rullistorna

column_aliases = {
    'headline': 'Rubrik',
    'number_of_vacancies': 'Antal Lediga Platser',
    'description.text': 'Beskrivning',
    'working_hours_type.label': 'Tidsomfattning',
    'workplace_address.region': 'Region',
    'workplace_address.municipality': 'Kommun'}


places_list = subset['workplace_address.region'].dropna().unique().tolist()
places_list.insert(0, 'Visa alla')


time_of_work = subset['working_hours_type.label'].dropna().unique().tolist()
time_of_work.insert(0, 'Visa alla')

# Select only these columns for initial display
ny_subset = subset[['headline', 'employer.workplace', 'description.text']]

# Display the DataFrame
st.subheader('Lediga jobb')

selected_place = st.selectbox("Välj region:", places_list)
selected_time_of_work = st.selectbox("Välj tidsomfattning:", time_of_work)

if selected_place == 'Visa alla':
    region_condition = subset['workplace_address.region'].notna()
else:
    region_condition = subset['workplace_address.region'] == selected_place

if selected_time_of_work == 'Visa alla':
    time_of_work_condition = subset['working_hours_type.label'].notna()
else:
    time_of_work_condition = subset['working_hours_type.label'] == selected_time_of_work

filtered_subset = subset[(region_condition) & (time_of_work_condition)]

for i in range(min(len(filtered_subset), 10)):
    with st.expander(f"Jobbannons {i+1} - {filtered_subset['headline'].iloc[i]}"):
        st.write("-------------------------------------------------")
        # Anropa OpenAI för att omformulera beskrivningstexten
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Du är expert på att skriva snygga jobbannonser"},
                {"role": "user", "content": filtered_subset['description.text'].iloc[i]},
            ]
        )

        # Hämta och skriv ut den genererade omformulerade beskrivningen
        for choice in response.choices:
            simplified_description = choice.message.content
            st.write(f"{simplified_description}")

selected_ads = st.multiselect("Välj annonser att visa detaljer för:", ny_subset['headline'])

#TEST SLUTAR HÄR
#
#





st.write(filtered_subset)


st.markdown("---")
st.subheader("Bakgrund till vårt projekt")
st.markdown("I vårt projekt...")
