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
with open('dataset.jsonl', 'r') as file: 
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

# Streamlit app
st.title('Jobtech Dataset')
# Display the DataFrame
st.write(subset)

# Display description of a specific row
row_index = st.slider("Välj radindex", 0, len(subset)-1, 25)
st.subheader("Beskrivning för vald rad:")
description_text = subset['description.text'].iloc[row_index]

"""# Uppmana OpenAI att generera en mer vardaglig beskrivning av beskrivningstexten
response = openai.Completion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Du är expert på att skriva snygga jobbannonser"},
        {"role": "user", "content": "Kan du skriva om jobbannonsen på ett mer flytande och enkelt sätt?"}
    ]
)"""

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Du är expert på att skriva snygga jobbannonser"},
    {"role": "user", "content": "Kan du skriva om jobbannonsen på ett mer flytande och enkelt sätt i en sammanhängande text?"},
  ]
)

# Hämta och skriv ut den genererade beskrivningen
simplified_description = response.choices[0].message
st.write(simplified_description)

# Show the variables in the dataset (equivalent to column names)
st.write("Columns in the dataset:")
st.write(subset.columns)
st.write('')

# Extract unique values from the column "employer.workplace"
places_list = subset['workplace_address.region'].unique().tolist()

# Display the list of different places jobs are located
st.write("List of Different Places Jobs are Located:")
st.write(places_list)

# Extract unique values from the column "working_hours_type.label"
time_of_work = subset['working_hours_type.label'].unique().tolist()
st.write("List of working hours type:")
st.write(time_of_work)

subset = subset.dropna(subset=['working_hours_type.label'])
subset['working_hours_type.label'] = subset['working_hours_type.label'].apply(lambda text: text.strip())

word_to_count = "deltid" 
try:
    subset['word_count'] = subset['working_hours_type.label'].apply(lambda text: text.lower().count(word_to_count.lower()))
    total_word_count = subset['word_count'].sum()
    st.write("\nTotal occurrences of '{}': {}".format(word_to_count, total_word_count))
    st.write(' ')
except Exception as e:
    st.write("An error occurred:", e)

# Först se till att alla rader som har NaN-värden i kolumnen 'working_hours_type.label' tas bort
subset = subset.dropna(subset=['working_hours_type.label'])
# Filtrera rader där kolumnen 'working_hours_type.label' innehåller ordet 'deltid'
deltid_rows = subset[subset['working_hours_type.label'].str.contains('deltid', na=False, case=False)]
# Skriv ut de filtrerade raderna
st.write(deltid_rows)
