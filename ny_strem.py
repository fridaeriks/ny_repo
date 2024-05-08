import pandas as pd
import json
import streamlit as st
import openai
from openai import OpenAI
import os
import requests
import io
import zipfile
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

print("Running...")

# Försök hämta data med maximalt 3 försök
for attempt in range(3):
    try:
        # Hämta data från URL och skriv till JSONL-fil
        if not os.path.exists("dataset.jsonl"):
            url = "https://data.jobtechdev.se/annonser/historiska/2023.jsonl.zip"

            zip_response = requests.get(url)
            zip_response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_file:
                file_name = zip_file.namelist()[0]
                with zip_file.open(file_name) as jsonl_file:
                    print("File loaded successfully!")

                    # Skriv till JSONL-filen
                    with open("dataset.jsonl", "wb") as output_file:  # Använd "wb"-läge för att skriva
                        for line in jsonl_file:
                            output_file.write(line)
                    print("JSONL data written successfully!")
        break  #Avsluta om försöket att hämta data är framgångsrikt
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

# Kontrollera om CSV-filen finns
if os.path.isfile('subset.csv'):
    # Om CSV-filen finns, läs in den i DataFrame
    subset = pd.read_csv('subset.csv')
else:
    # Ladda upp JASONL filen in i vår DataFrame
    lines = []
    try:
        url = "https://data.jobtechdev.se/annonser/historiska/2023.jsonl.zip"
        zip_response = requests.get(url)
        zip_response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_file:
            file_name = zip_file.namelist()[0]
            with zip_file.open(file_name) as jsonl_file:
                print("File loaded successfully!")

                # Konvertera varje rad från JSON-format till Python-dictionary
                for i, line in enumerate(jsonl_file):
                    lines.append(line.strip())
                    if i >= 9999:
                        break


        # Konvertera varje rad från JSON-format till Python-dictionary
        data = [json.loads(line) for line in lines]

        # Om JSON-filen har nästlade strukturer, kommer pandas automatiskt att platta ut dem
        jobtech_dataset = pd.json_normalize(data)

        # De komunder som ingår i datasettet
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

        # Gör om subdet DataFrame till CSV
        subset.to_csv('subset.csv', index=False)

        print("Done!")

    except requests.exceptions.RequestException as e:
        print("Failed to fetch data:", e)
    except Exception as e:
        print("Error:", e)

print("Almost done!")


# Ladda in nltk:s stemmingfunktion för svenska
from nltk.stem.snowball import SnowballStemmer
stemmer_sv = SnowballStemmer("swedish")

# Ladda in stoppord för svenska från nltk
from nltk.corpus import stopwords
stop_words_sv = set(stopwords.words('swedish'))

# Ladda in engelska stoppord för att hantera engelska texter
stop_words_en = set(stopwords.words('english'))

# Ladda in punktuation från string
import string
punctuation = set(string.punctuation)

# Select only relevant columns and make a copy to avoid SettingWithCopyWarning
new_subset = subset[[
    'headline',
    'description.text'
]].copy()

# Define a function to preprocess text with stemming for both English and Swedish
def preprocess_text(text, language='english'):
    # Convert list of keywords to a single string and then convert to lowercase
    if isinstance(text, list):
        text = ' '.join(text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words and punctuation based on language
    if language == 'swedish':
        stop_words = stop_words_sv
        stemmer = stemmer_sv
    else:
        stop_words = stop_words_en
        stemmer = SnowballStemmer("english")
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
    # Stem the tokens
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    # Join the stemmed tokens back into a single string
    text = ' '.join(stemmed_tokens)
    return text

# Apply text preprocessing with stemming
#new_subset['combined_text'] = new_subset.apply(lambda row: preprocess_text(row['headline'] + ' ' + row['description.text'], language='swedish'), axis=1)
new_subset['combined_text'] = new_subset.apply(lambda row: preprocess_text((row['headline'] if pd.notnull(row['headline']) else '') + ' ' + (row['description.text'] if pd.notnull(row['description.text']) else ''), language='swedish'), axis=1)


# Justera vektoriseringen för att använda olika parametrar
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(new_subset['combined_text'])

# Optimera antalet kluster med elbow method eller silhouette score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

max_clusters = 10  # Justera detta beroende på dina data
silhouette_scores = []
for num_clusters in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plotta silhouette score för olika antal kluster
plt.plot(range(2, max_clusters + 1), silhouette_scores)
plt.xlabel('Antal kluster')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score för olika antal kluster')
plt.show()

# Antal kluster, inklusive "Lagerarbete och logistik" och "Övrigt"
optimal_num_clusters = 7  # Justera detta baserat på din analys

# Använd det optimala antalet kluster för att utföra klustringen
kmeans = KMeans(n_clusters=optimal_num_clusters)
kmeans.fit(X)

# Manuellt tilldela namn till varje kluster baserat på de framträdande orden eller nyckelorden
cluster_names = [
    "Teknologi och IT",
    "Hälsovård och medicin",
    "Utbildning och pedagogik",
    "Ekonomi och finans",
    "Försäljning och marknadsföring",
    "Lagerarbete och logistik",
    "Övrigt"
]

# Lägg till en ny kolumn i DataFrame för branschnamn
subset['industry'] = [cluster_names[label] for label in kmeans.labels_]


# Titel och text högst upp
st.markdown("<h1 style='color: red; display: inline;'>ATH</h1><h1 style='color: black; display: inline;'>WORK</h1>", unsafe_allow_html=True)
=======
# Titel och text högst upp 
st.markdown("<h1 style='color: red; display: inline;'><span style='font-style: italic;'>ATH</span></h1><h1 style='color: black; display: inline;'>WORK</h1>", unsafe_allow_html=True)


st.markdown("Info om vårt projekt")
st.markdown("---")
st.write(subset)



# Grå sidopanel

# Text som ingår i vidare läsning (sidopanel)
vidare_lasning = """Text om vi vill ha...

[Swedish Elite Sport](https://www.idan.dk/media/stgjthhj/swedish-elite-sport.pdf) handlar om de 
ekonomiska utmaningarna för svenska idrottare i jämförelse med Norge och Danmark, 
där texten ppekar på bristande stöd under utvecklingsfasen och den resulterande ekonomiska osäkerheten.

[How 5 Athletes Afford to Stay in the Game and Still Make Rent](https://www.thecut.com/2024/01/pro-athletes-working-second-jobs-careers.html) 
handlar om hur idrottare, särskilt kvinnor och de i mindre populära idrottsgrenar, 
kämpar med ekonomisk osäkerhet och måste kombinera sin idrottskarriär med andra jobb 
för att klara ekonomin."""

# Text som ingår i kontaktuppgifter (sidopanel)
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

# Text som ingår i om oss (sidopanel)
om_oss = '...'

# Text som ingår i bakgrund (sidopanel)
bakgrund = '...'

left_column = st.sidebar


# Texten i sidopanelen: annan text som vi kan lägga till
left_column.markdown("### Fri text")
left_column.markdown("Text...")

# Vidare läsning i sidopanelen
with left_column.expander("📖 Vidare läsning"):
    st.write(vidare_lasning)

# Kontaktuppgifter i sidopanelen
with left_column.expander("📞 Kontaktuppgifter"):
    st.write(kontakt_uppgifter)

# Om oss i sidopanelen
with left_column.expander("👥 Om oss"):
    st.write(om_oss)

# Bakgrund i sidopanelen
with left_column.expander("📚 Projektets bakgrund"):
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


places_list = subset['workplace_address.region'].dropna().unique().tolist()
places_list.insert(0, 'Visa alla')


time_of_work = subset['working_hours_type.label'].dropna().unique().tolist()
time_of_work.insert(0, 'Visa alla')

duration_time = subset['duration.label'].dropna().unique().tolist()
duration_time.insert(0, 'Visa alla')



# Visa DataFrame
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

# Add a selectbox for industry sectors
selected_industry = st.selectbox("Välj bransch:", ['Visa alla'] + cluster_names)

# Update filtering logic to include selected industry
if selected_industry == 'Visa alla':
    industry_condition = subset['industry'].notna()
else:
    industry_condition = subset['industry'] == selected_industry

# Filtered subset based on all conditions
filtered_subset = subset[(region_condition) & (time_of_work_condition) & (duration_condition) & (industry_condition)]


filtered_subset = filtered_subset[['headline', 'employer.workplace', 'number_of_vacancies', 'description.text', 
                                   'working_hours_type.label', 'workplace_address.region', 
                                   'workplace_address.municipality', 'duration.label', 'industry']]

filtered_subset = filtered_subset.rename(columns=column_aliases) 

#FRIDAS ÄNDRING START


#FRIDAS ÄNDRING SLUT


job_count = filtered_subset.shape[0]

st.markdown(f"<h1 style='font-weight: bold; color: green;'>{job_count} st </h1>", unsafe_allow_html=True)
st.markdown("jobb matchar din sökning")


# Välj endast dessa tre kolumner
ny_subset = filtered_subset[[
    'headline',
    'employer.workplace',  
    'description.text'
]]

# Titel och text högst upp
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
                    {"role": "system", "content": """Du är expert på att skriva snygga jobbannonser 
                     och alla jobbanonser ska vara skrivna på samma sätt det vill säga med liknande rubriker och innehåll utan listor.
                     """},
                    {"role": "user", "content": filtered_subset['description.text'].iloc[i]},
                ]
            )

            #Hämta och skriv ut den genererade omformulerade beskrivningen
            for choice in response.choices:
                simplified_description = choice.message.content
                st.write(f"{simplified_description}")



#visa fler alternativ
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

                    #Hämta och skriv ut den genererade omformulerade beskrivningen
                    for choice in response.choices:
                        simplified_description = choice.message.content
                        st.write(f"{simplified_description}")
                  
#selected_ads = st.multiselect("Välj annonser att visa detaljer för:", ny_subset['Rubrik'])

st.write(filtered_subset)


# Text längst ner på sidan
st.markdown("---")
st.subheader("Bakgrund till vårt projekt")
st.markdown("I vårt projekt...")
