import pandas as pd
import json
import streamlit as st
import openai
from openai import OpenAI
import os
import requests
import io
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


#--------------------------------------------------------------------------------------------------------------------------#
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

#--------------------------------------------------------------------------------------------------------------------------#

API_KEY = open('Open_AI_key', 'r').read()

client = OpenAI(
    api_key=API_KEY
) 

# Läs in API-nyckeln från filen
with open("Open_AI_key", "r") as file:
    api_key = file.read().strip()

# Ange din API-nyckel
openai.api_key = api_key


#--------------------------------------------------------------------------------------------------------------------------#

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

#--------------------------------------------------------------------------------------------------------------------------#

import nltk
print(nltk.data.path)

#nltk.data.path.append("/path/to/nltk_data")
nltk.download('stopwords')

# Ladda in nltk:s stemmingfunktion för svenska
from nltk.stem.snowball import SnowballStemmer
stemmer_sv = SnowballStemmer("swedish")

# Ladda in stoppord för svenska från nltk
from nltk.corpus import stopwords
stop_words_sv = set(stopwords.words('swedish'))

# Ladda in engelska stoppord för att hantera engelska texter
stop_words_en = set(stopwords.words('english'))

nltk.download('stopwords')

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

#--------------------------------------------------------------------------------------------------------------------------#

#Miranda uppdatering 1
st.markdown("<h1 style='color: red; display: inline;'>ATH</h1><h1 style='color: black; display: inline;'>WORK</h1>", unsafe_allow_html=True)
st.markdown("Det ska vara lätt att hitta jobb för just dig!")

st.markdown("---")

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

with left_column.expander("👥 Om oss"):
    st.write(om_oss)

# Vidare läsning i sidopanelen
with left_column.expander("📖   Vidare läsning"):
    st.write(vidare_lasning)

# Kontaktuppgifter i sidopanelen
with left_column.expander("📞   Kontaktuppgifter"):
    st.info(kontakt_uppgifter)


# Bakgrund i sidopanelen
with left_column.expander("📚   Projektets bakgrund"):
    st.write(bakgrund) 


#--------------------------------------------------------------------------------------------------------------------------#

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

places_list = subset['workplace_address.region'].dropna().unique().tolist()
places_list.insert(0, 'Visa alla')


time_of_work = subset['working_hours_type.label'].dropna().unique().tolist()
time_of_work.insert(0, 'Visa alla')

duration_time = subset['duration.label'].dropna().unique().tolist()
duration_time.insert(0, 'Visa alla')

# Visa DataFrame
st.subheader('Lediga jobb')

search_query = st.text_input('Sök efter specifika ord:', value="", help="Jobbtitel, nyckelord eller företag etc",)

col1, col2, col3, col4 = st.columns(4)

with col1:
   selected_place = st.selectbox(f'Välj region:', places_list)

with col2:
   selected_time_of_work = st.selectbox(f'Välj anställningsform:', time_of_work)

with col3:
   selected_duration_time = st.selectbox(f'Välj tidsomfattning', duration_time)

with col4:
    # Add a selectbox for industry sectors
    selected_industry = st.selectbox("Välj bransch:", ['Visa alla'] + cluster_names)

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

if search_query:
    text_condition = df['description.text'].str.contains(search_query, case=False, na=False)
else:
    text_condition = pd.Series(True, index=df.index)  # Default condition to select all rows

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

#--------------------------------------------------------------------------------------------------------------------------#

job_count = filtered_subset.shape[0]

#--------------------------------------------------------------------------------------------------------------------------#

#Visar hur många lediga jobba som finns
st.markdown(f"<h1 style='font-weight: bold; color: green;'>{job_count} st </h1>", unsafe_allow_html=True)
st.markdown("Jobb som matchar sökningen:")


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

st.markdown("---")                  
#--------------------------------------------------------------------------------------------------------------------------#

#SUPERVISED LEARNING
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Ladda ned stoppord och lexikon för lemmatisering
nltk.download('stopwords')
nltk.download('punkt')


# Läs in data
df = pd.read_csv('subset.csv').head(206)
# Läs in 'Headline' från CSV-filen
pd.read_csv('subset.csv')['headline'].head(206)


# Skapa en kopia av den ursprungliga kolumnen
df['stemmed_text'] = df['description.text']

# Definiera stoppord
swedish_stop_words = set(stopwords.words('swedish'))
# Skapa en instans av SnowballStemmer för svenska
stemmer = SnowballStemmer('swedish')

# Funktion för textpreprocessering för en specifik kolumn
def preprocess_text_column(column):
    # Tokenisera texten i kolumnen
    column_tokens = [word_tokenize(text.lower(), language='swedish') for text in column]
    # Ta bort stoppord och ord med en längd mindre än 3, samt stamma ord
    preprocessed_column = []
    for tokens in column_tokens:
        filtered_tokens = [stemmer.stem(token) for token in tokens if token not in swedish_stop_words and len(token) > 2 and token.isalpha()]
        preprocessed_column.append(' '.join(filtered_tokens))
    
    return preprocessed_column


# Preprocessa texten i kolumnen 'description.text'
df['stemmed_text'] = preprocess_text_column(df['stemmed_text'])


# Funktion för att extrahera viktiga ord från jobbannonser
def extract_manual_keywords():
    # Lista över manuellt valda viktiga ord
    manual_keywords = ["tävlingsinriktad", "35 timmar", "flexibelt arbete", "deltid", "extra personal"]
    
    return manual_keywords
# Extrahera de manuellt valda viktiga orden
manual_keywords = extract_manual_keywords()
# Lägg till de manuellt valda viktiga orden i vokabulären för TF-IDF-vektoriseringen
vectorizer = TfidfVectorizer(vocabulary=manual_keywords)


# Manuellt märkta etiketter för de första 200 raderna
labels = ["NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA",
          "NEJ", "JA", "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", "JA", "JA",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", 
          "NEJ", "JA", "JA", "JA", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "JA", "NEJ", "JA", "NEJ", "JA", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "JA", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "JA", "JA", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "JA", "JA", "JA", "JA",
          "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA",
          "JA", "NEJ", "NEJ", "JA", "JA", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "JA", "NEJ", "JA", "NEJ", 
          "NEJ", "JA", "NEJ", "JA", "NEJ", "NEJ", "NEJ", "JA", "JA", "NEJ", "NEJ", 
          "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "JA", "JA", "NEJ", "NEJ", "NEJ", 
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", 
          "NEJ", "NEJ", "JA", "JA", "NEJ", "JA", "JA", "NEJ"]

# Skapa en ny kolumn med namnet "label" och tilldela den dina manuellt märkta etiketter
df['label'] = labels[:len(df)]
# Ta bara de första 200 raderna som har en etikett
df_with_labels = df.dropna(subset=['label']).head(200)

# Förutsatt att ditt dataset finns i en DataFrame df med kolumnen "description.text" för jobbannonserna och "label" för etiketten
X = df_with_labels['stemmed_text']
y = df_with_labels['label']

# Dela upp data i träningsdata och testdata
# Dela upp data i träningsdata (150) och testdata (50) slumpmässigt
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=120, test_size=80, random_state=42)
# Skapa TF-IDF-vektorer från text med samma vokabulär som användes för träning
X_train_vectorized = vectorizer.fit_transform(X_train)
# Använd samma vektoriseringsinstans för att transformera testdatan
X_test_vectorized = vectorizer.transform(X_test)

# Välj modell (Logistisk regression) och ange viktade klasser
model = LogisticRegression(max_iter=1000, class_weight={'NEJ': 1, 'JA': 10})

# Träna modellen
model.fit(X_train_vectorized, y_train)
# Förutsäg på testdata
y_pred = model.predict(X_test_vectorized)
# Utvärdera modellens prestanda
print(classification_report(y_test, y_pred))
# Förutsäg lämpligheten för varje jobbannons i ditt dataset
df['prediction'] = model.predict(vectorizer.transform(df['stemmed_text']))
# Sortera DataFrame baserat på förutsägelserna för att få jobbannonserna i kronologisk ordning för vad som passar bäst med idrott
sorted_df = df.sort_values(by='prediction', ascending=False)

st.subheader("AI-generator")
info = """Nedan listar en AI de tre bäst lämpade arbeten för elitidrottare. Dessa förslag har utvecklats utifrån en supervised model som tränats för att ge bästa möjliga rekommendation.

Detta är endast en prototyp och inte en färdigt utvecklad modell.

###Top tre:"""

st.write(info)
top_predictions = sorted_df[['headline','description.text', 'prediction']].head(3)



for i in range(len(top_predictions)):
        with st.expander(f"{top_predictions['headline'].iloc[i]}"):
            st.write("-------------------------------------------------")
            # Anropa OpenAI för att omformulera beskrivningstexten
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """Du är expert på att skriva snygga jobbannonser 
                     och alla jobbanonser ska vara skrivna på samma sätt det vill säga med liknande rubriker och innehåll utan listor.
                     """},
                    {"role": "user", "content": top_predictions['description.text'].iloc[i]},
                ]
            )

            #Hämta och skriv ut den genererade omformulerade beskrivningen
            for choice in response.choices:
                simplified_description = choice.message.content
                st.write(f"{simplified_description}")


#--------------------------------------------------------------------------------------------------------------------------#
# Text längst ner på sidan
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

