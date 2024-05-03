
import pandas as pd
import json
import streamlit as st
import openai
from openai import OpenAI
import os
import requests
import io
import zipfile

print("Running...")

# Retry fetching data with a maximum of 3 attempts
for attempt in range(3):
    try:
        # Fetch data from URL and write to JSONL file
        if not os.path.exists("dataset.jsonl"):
            url = "https://data.jobtechdev.se/annonser/historiska/2023.jsonl.zip"

            zip_response = requests.get(url)
            zip_response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_file:
                file_name = zip_file.namelist()[0]
                with zip_file.open(file_name) as jsonl_file:
                    print("File loaded successfully!")

                    # Write to the JSONL file
                    with open("dataset.jsonl", "wb") as output_file:  # Use "wb" mode to write
                        for line in jsonl_file:
                            output_file.write(line)
                    print("JSONL data written successfully!")
        break  # Exit the retry loop if data fetching is successful
    except requests.exceptions.RequestException as e:
        print("Failed to fetch data (attempt {}): {}".format(attempt + 1, e))
    except Exception as e:
        print("Error:", e)

API_KEY = open('Open_AI_key', 'r').read()

client = OpenAI(
    api_key=API_KEY
) 

# Läs in API-nyckeln från filen
with open("Open_AI_key", "r") as file:
    api_key = file.read().strip()

# Ange din API-nyckel
openai.api_key = api_key

# Check if the CSV file exists
if os.path.isfile('subset.csv'):
    # If CSV exists, read it into DataFrame
    subset = pd.read_csv('subset.csv')
else:
    # Load the JSON file into a DataFrame 
    lines = []
    try:
        url = "https://data.jobtechdev.se/annonser/historiska/2023.jsonl.zip"
        zip_response = requests.get(url)
        zip_response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_file:
            file_name = zip_file.namelist()[0]
            with zip_file.open(file_name) as jsonl_file:
                print("File loaded successfully!")

                # Convert each line from JSON format to Python dictionary
                for i, line in enumerate(jsonl_file):
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

        # Write the subset DataFrame to CSV
        subset.to_csv('subset.csv', index=False)

        print("Done!")

    except requests.exceptions.RequestException as e:
        print("Failed to fetch data:", e)
    except Exception as e:
        print("Error:", e)

print("Almost done!")

#Titel och text högst upp
st.markdown("<h1 style='color: red; display: inline;'>ATH</h1><h1 style='color: black; display: inline;'>WORK</h1>", unsafe_allow_html=True)

st.markdown("Info om vårt projekt")
st.markdown("---")
st.write(subset)



#Den gråa sidopanelen
vidare_lasning = """Text om vi vill ha...

[Swedish Elite Sport](https://www.idan.dk/media/stgjthhj/swedish-elite-sport.pdf) handlar om de 
ekonomiska utmaningarna för svenska idrottare i jämförelse med Norge och Danmark, 
där texten ppekar på bristande stöd under utvecklingsfasen och den resulterande ekonomiska osäkerheten.

[How 5 Athletes Afford to Stay in the Game and Still Make Rent](https://www.thecut.com/2024/01/pro-athletes-working-second-jobs-careers.html) 
handlar om hur idrottare, särskilt kvinnor och de i mindre populära idrottsgrenar, 
kämpar med ekonomisk osäkerhet och måste kombinera sin idrottskarriär med andra jobb 
för att klara ekonomin."""

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




#Tabell där man kan filtrera med båda rullistorna

column_aliases = {
    'headline': 'headline',
    'employer.workplace': 'employer.workplace',
    'number_of_vacancies': 'number_of_vacancies',
    'description.text': 'description.text',
    'working_hours_type.label': 'working_hours_type.label',
    'workplace_address.region': 'workplace_address.region',
    'workplace_address.municipality': 'workplace_address.municipality',
    'duration.label': 'duration.label'
}



places_list = subset['workplace_address.region'].dropna().unique().tolist()
places_list.insert(0, 'Visa alla')


time_of_work = subset['working_hours_type.label'].dropna().unique().tolist()
time_of_work.insert(0, 'Visa alla')

duration_time = subset['duration.label'].dropna().unique().tolist()
duration_time.insert(0, 'Visa alla')

# Select only these columns for initial display
ny_subset = subset[['headline', 'employer.workplace', 'description.text']]

# Display the DataFrame
st.subheader('Lediga jobb')

selected_place = st.selectbox("Välj region:", places_list)
selected_time_of_work = st.selectbox("Välj anställningsform:", time_of_work)
selected_duration_time = st.selectbox(f'Välj tidsomfattning', duration_time)


if selected_place == 'Visa alla':
    region_condition = subset['workplace_address.region'].notna()
else:
    region_condition = subset['workplace_address.region'] == selected_place

if selected_time_of_work == 'Visa alla':
    time_of_work_condition = subset['working_hours_type.label'].notna()
else:
    time_of_work_condition = subset['working_hours_type.label'] == selected_time_of_work

if selected_duration_time == 'Visa alla':
    duration_condition = subset['duration.label'].notna()
else:
    duration_condition = subset['duration.label'] == selected_duration_time


filtered_subset = subset[(region_condition) & (time_of_work_condition) & (duration_condition)]

filtered_subset = filtered_subset[['headline', 'employer.workplace', 'number_of_vacancies', 'description.text', 
                                   'working_hours_type.label', 'workplace_address.region', 
                                   'workplace_address.municipality', 'duration.label']]

filtered_subset = filtered_subset.rename(columns=column_aliases) 

#HÄR HAR JAG ÄNDRAT

job_count = filtered_subset.shape[0]

st.markdown(f"<h1 style='font-weight: bold; color: green;'>{job_count} st </h1>", unsafe_allow_html=True)
st.markdown("jobb matchar din sökning")




###



# Select only these columns
ny_subset = filtered_subset[[
    'headline',
    'employer.workplace',  
    'description.text'
]]

# Title and text at the top
st.subheader('Lediga jobb')




number = 2
 
temp = st.empty()

with temp.container():
    for i in range(min(len(ny_subset), number)):
        with st.expander(f"Jobbannons {i+1} - {ny_subset['headline'].iloc[i]}"):
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



#Show more options
if len(ny_subset) > number:
    if st.button('Visa fler'):
        temp.empty()
        number += 2
        temp = st.empty()
        with temp.container():
            for i in range(number - 2, min(len(ny_subset), number)):
                with st.expander(f"Jobbannons {i+1} - {ny_subset['headline'].iloc[i]}"):
                    st.write("-------------------------------------------------")
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
                  


                




#selected_ads = st.multiselect("Välj annonser att visa detaljer för:", ny_subset['Rubrik'])







st.write(filtered_subset)


st.markdown("---")
st.subheader("Bakgrund till vårt projekt")
st.markdown("I vårt projekt...")
