import streamlit as st
import pandas as pd
import json
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

# Antag att du redan har subset och column_aliases definierade
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

# För att kombinera de två kodsnuttarna:
# Visa tabellen där användare kan filtrera med båda rullistorna
column_aliases = {
    'headline': 'Rubrik',
    'number_of_vacancies': 'Antal Lediga Platser',
    'description.text': 'Beskrivning',
    'working_hours_type.label': 'Tidsomfattning',
    'workplace_address.region': 'Region',
    'workplace_address.municipality': 'Kommun'}

places_list = subset['workplace_address.region'].unique().tolist()
time_of_work = subset['working_hours_type.label'].unique().tolist()
selected_place = st.selectbox("Select Region:", places_list)
selected_time_of_work = st.selectbox("Select Time of Work:", time_of_work)

filtered_subset = subset[(subset['workplace_address.region'] == selected_place) & 
                         (subset['working_hours_type.label'] == selected_time_of_work)]

filtered_subset = filtered_subset[['headline', 'number_of_vacancies', 'description.text', 
                                   'working_hours_type.label', 'workplace_address.region', 
                                   'workplace_address.municipality']]

filtered_subset = filtered_subset.rename(columns=column_aliases) 
st.write(filtered_subset)

# Lägg till kod för att förbättra texten i kolumnen "Rubrik" med AI
# Loopa igenom varje rad i den filtrerade tabellen
for index, row in filtered_subset.iterrows():
    # Om det inte finns någon rubrik, använd beskrivningen för att generera en
    if pd.isnull(row['Rubrik']):
        # Generera en förenklad rubrik med AI-modellen baserat på beskrivningstexten
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Du är expert på att skriva tydliga, snygga titlar till jobbannonser"},
                {"role": "user", "content": row['Beskrivning']},  # Använd beskrivningstexten som input
            ]
        )
        # Extrahera den genererade rubriken från AI-modellen
        simplified_headline = response.choices[0].message.content
        
        # Uppdatera den aktuella raden i den filtrerade tabellen med den genererade rubriken
        filtered_subset.at[index, 'Rubrik'] = simplified_headline

# Visa den uppdaterade tabellen med de förbättrade rubrikerna
st.write(filtered_subset)
