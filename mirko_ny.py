import pandas as pd
import json
import streamlit as st
import openai
from openai import OpenAI
import os
import streamlit_shadcn_ui as ui
import requests
import io
import zipfile
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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


# Define the hover card inside the main content area
ui.hover_card(label="För arbetsgivare", content="För arbetsgivare", content_type="text", key="hover_card_1")


#Fråga oss pratbubbla
st.markdown(
    """
    <div style="position: fixed; bottom: 20px; right: 20px; width: 90px; height: 40px; background-color: rgba(240, 240, 240, 0.8); border-radius: 10px; padding: 10px; display: flex; justify-content: center; align-items: center;">
        <div style="position: absolute; top: 50%; left: 100%; margin-top: -10px; width: 0; height: 0; border-top: 10px solid transparent; border-bottom: 10px solid transparent; border-left: 10px solid rgba(240, 240, 240, 0.8);"></div>
        <p style="margin: 0; color: #333;">Fråga oss!</p>
    </div> 
    """,
    unsafe_allow_html=True
)



st.image('logo2.jpg', width=180)  


#Titel och text högst upp 
st.markdown("Det ska vara lätt att hitta jobb för just dig!")
st.markdown("---")
#st.write(subset)

#Den gråa sidopanelen
om_oss = (f'Vårt projekt arbete hamdlar om... Ett stort problem har upptäckts.... Vill lösa detta... Genom intervjuer etc...')

vidare_lasning = """Text om vi vill ha...

[Swedish Elite Sport](https://www.idan.dk/media/stgjthhj/swedish-elite-sport.pdf) handlar om de 
ekonomiska utmaningarna för svenska idrottare i jämförelse med Norge och Danmark, 
där texten ppekar på bristande stöd under utvecklingsfasen och den resulterande ekonomiska osäkerheten.

[How 5 Athletes Afford to Stay in the Game and Still Make Rent](https://www.thecut.com/2024/01/pro-athletes-working-second-jobs-careers.html) 
handlar om hur idrottare, särskilt kvinnor och de i mindre populära idrottsgrenar, 
kämpar med ekonomisk osäkerhet och måste kombinera sin idrottskarriär med andra jobb 
för att klara ekonomin."""

kontakt_uppgifter = """
Python Consulant 
Vera Hertzman
Vera@outlook.com
+46 76 848 23 49

Head of AI 
Thea Håkansson
Thea@gmail.se
+46 73 747 87 45

Head of Streamlit 
Frida Eriksson
Royal@yahoo.com
+46 76 432 38 49

Project Coordinator 
Miranda Tham
Miranda@hotmail.com
+46 76 767 00 35

Agenda and Report Administrator Tove Lennertsson
Tove@gmail.com
+46 0000000"""

bakgrund = """Här kommer info om projektets bakgrund """

left_column = st.sidebar.container()

left_column.write("""
<style>
.left-column {
    background-color: #FF7F7F;
    width: 30%;
    padding: 20px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

                    #Texten i sidopanelen: annan text som vi kan lägga till
left_column.markdown("### Vi på ATH work")
                     #left_column.markdown("Info om vårt projekt")

                    #Vidare läsning i sidopanelen

with left_column.expander("💼 Om oss"):
    st.write(om_oss)

# Vidare läsning i sidopanelen
with left_column.expander("📖   Vidare läsning"):
    st.write(vidare_lasning)

# Kontaktuppgifter i sidopanelen
with left_column.expander("📫   Kontaktuppgifter"):
    st.info(kontakt_uppgifter)


# Bakgrund i sidopanelen
with left_column.expander("📋   Projektets bakgrund"):
    st.write(bakgrund) 

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

df = pd.read_csv("subset.csv")

places_list = df['workplace_address.region'].dropna().unique().tolist()
places_list.insert(0, 'Visa alla')


time_of_work = df['working_hours_type.label'].dropna().unique().tolist()
time_of_work.insert(0, 'Visa alla')

duration_time = df['duration.label'].dropna().unique().tolist()
duration_time.insert(0, 'Visa alla')

# Display the DataFrame
st.subheader('Sök bland lediga jobb' )

search_query = st.text_input('Sök efter specifika ord:', value="", help="Jobbtitel, nyckelord eller företag etc",)

col1, col2, col3 = st.columns(3)

with col1:
   selected_place = st.selectbox(f'Välj region:', places_list, help="Län i Sverige")

with col2:
   selected_time_of_work = st.selectbox(f'Välj anställningsform:', time_of_work)

with col3:
   selected_duration_time = st.selectbox(f'Välj tidsomfattning', duration_time)


if selected_place == 'Visa alla':
    region_condition = df['workplace_address.region'].notna()
else:
    region_condition = df['workplace_address.region'] == selected_place

if selected_time_of_work == 'Visa alla':
    time_of_work_condition = df['working_hours_type.label'].notna()
else:
    time_of_work_condition = df['working_hours_type.label'] == selected_time_of_work

if selected_duration_time == 'Visa alla':
    duration_condition = df['duration.label'].notna()
else:
    duration_condition = df['duration.label'] == selected_duration_time

if search_query:
    text_condition = df['description.text'].str.contains(search_query, case=False, na=False)
else:
    text_condition = pd.Series(True, index=df.index)  # Default condition to select all rows

# Filter DataFrame based on all conditions
filtered_subset = df[(region_condition) & 
                     (time_of_work_condition) 
                     & (duration_condition) 
                     & (text_condition)
                     ]

job_count = filtered_subset.shape[0]

#Visar hur många lediga jobba som finns
st.markdown(f"<h1 style='font-weight: bold; color: green;'>{job_count} st </h1>", unsafe_allow_html=True)
st.markdown("Jobb som matchar sökningen:")


# Title and text at the top
st.subheader('Lediga jobb')

number = 2 
temp = st.empty()

with temp.container():
    for i in range(min(len(filtered_subset), number)):
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


ny_subset = filtered_subset[['headline', 
                             'employer.workplace',
                            'number_of_vacancies', 
                            'description.text' 
                             ]]
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
                
#Längst ner, bakgrund till vårt projekt
st.markdown("---")
st.subheader("Bakgrund till vårt projekt")
st.markdown("I vårt projekt...")



col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("<h9 style='text-align:'>Frida Eriksson</h9>", unsafe_allow_html=True)
    st.image('https://static.streamlit.io/examples/cat.jpg', width=100)

with col2:
    st.markdown("<h9 style='text-align:'>Miranda Tham</h9>", unsafe_allow_html=True)
    st.image('kat.jpg', width=100)

with col3:
    st.markdown("<h9 style='text-align:'>Thea Håkansson</h9>", unsafe_allow_html=True)
    st.image('kat.jpg', width=100)

with col4:
    st.markdown("<h9 style='text-align:'>Vera Hertzman</h9>", unsafe_allow_html=True)
    st.image('kat.jpg', width=100)

with col5:
    st.markdown("<h9 style='text-align: center;'>Tove Lennartson</h9>" , unsafe_allow_html=True)
    st.image('kat.jpg', width=100)


#Panelen längst ner

st.markdown(
    """
    <style>
        .line {
            width: 100%;
            height: 4px;
            background-color: black; /* Navy färg */
            margin-bottom: 20px;
        }
    </style>
    <div class="line"></div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<h6 style='text-align:left;'>Säkerhet</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Kundsäkerhet</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Hantering av kunduppgifter</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Falska mail</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Anmäl ett fel</h6>", unsafe_allow_html=True)
    

with col2:
    st.markdown("<h6 style='text-align:left;'>För föreingen</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Lägg till egen annons</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Ändra layout</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Visa alla jobb</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Inloggning för förenigar</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Administrera föreningsannonser</h6>", unsafe_allow_html=True)

with col3:
    st.markdown("<h6 style='text-align:left;'>Villkor</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Användarvillkor</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Personuppgiftshantering</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Cookies</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Cookiesinställningar</h6>", unsafe_allow_html=True)

with col4:
    st.markdown("<h6 style='text-align:left;'>SPORTEE</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Om SPORTEE</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Press</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Jobba på SPORTEE</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Kontakta oss</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Inspiration</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Tips och guider</h6>", unsafe_allow_html=True)


