import streamlit as st
import pandas as pd
import json
from openai import OpenAI
import openai

API_KEY = open('Open_AI_key', 'r').read()

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
    'headline': 'Rubrik',  # Använd 'headline' för rubriken här
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



# Lista för att lagra de genererade rubrikerna
generated_headlines = []

# Kontroll om den filtrerade tabellen är tom
if not filtered_subset.empty:
    # Loopa igenom varje rad i den filtrerade tabellen
    for index, row in filtered_subset.iterrows():
        # Om det inte finns någon rubrik, använd beskrivningen för att generera en
        if pd.isnull(row['Rubrik']):
            # Generera en förenklad rubrik med AI-modellen baserat på beskrivningstexten
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Du är expert på att skriva rubriker till jobbannonser baserat på jobbeskrivningen"},
                    {"role": "user", "content": row['Beskrivning']},  # Använd beskrivningstexten som input
                ]
            )
            # Extrahera den genererade rubriken från AI-modellen
            simplified_headline = response.choices[0].message.content
            
            # Lägg till den genererade rubriken i listan
            generated_headlines.append(simplified_headline)

            # Utskrift för att kontrollera den genererade rubriken
            print("Rad", index, "Genererad rubrik:", simplified_headline)
        else:
            # Om rubriken redan finns, lägg till den i listan
            generated_headlines.append(row['Rubrik'])

    # Uppdatera den filtrerade tabellen med de genererade rubrikerna
    filtered_subset['Rubrik'] = generated_headlines

# Visa den uppdaterade tabellen med de förbättrade rubrikerna
st.write(filtered_subset)


       


#Du är expert på att skriva rubriker till jobbannonser baserat på jobbeskrivningen