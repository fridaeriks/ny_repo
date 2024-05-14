import pandas as pd
import streamlit as st
import openai
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem.snowball import SnowballStemmer

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


#--------------------------------------------------------------------------------------------------------------------------#
print("Running...")
#Vår logga
st.image('logo2.jpg', width=300)  

#Den gråa sidopanelen
st.markdown("Det ska vara lätt att hitta jobb för just dig!")
st.markdown("---")

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)


om_oss = (f'Vårt projekt arbete hamdlar om... Ett stort problem har upptäckts.... Vill lösa detta... Genom intervjuer etc...')

vidare_lasning = """
Rapporten Swedish Elite Sport handlar om de svenska idrottarnas ekonomiska utmaningar, i jämförelse med våra grannar Norge och Danmark. 
Texten pekar på ett bristande svenskt idrottsstöd under utvecklingsfasen som har resulterat i den nuvarande ekonomiska osäkerheten 
hos våra svenska idrottare.
[Läs mer](https://www.idan.dk/media/stgjthhj/swedish-elite-sport.pdf)

How 5 Athletes Afford to Stay in the Game and Still Make Rent är en amerikansk artikel som handlar om hur idrottare, 
särskilt kvinnor och i de mindre populära idrottsgrenarna, globalt sett lever i en ekonomisk kamp och osäkerhet.
[Läs mer](https://www.thecut.com/2024/01/pro-athletes-working-second-jobs-careers.html)"""

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

#Vänstra kolumnen
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
left_column.markdown("### Vi på <span style='color: #4a90e2;'>SPORTEE</span>", unsafe_allow_html=True)
                     #left_column.markdown("Info om vårt projekt")

                    #Vidare läsning i sidopanelen

with left_column.expander("💼 Om oss"):
    st.info(om_oss)

# Vidare läsning i sidopanelen
with left_column.expander("📖   Vidare läsning"):
    st.info(vidare_lasning)

# Kontaktuppgifter i sidopanelen
with left_column.expander("📫   Kontaktuppgifter"):
    st.info(kontakt_uppgifter)


# Bakgrund i sidopanelen
with left_column.expander("📋   Projektets bakgrund"):
    st.info(bakgrund) 
#--------------------------------------------------------------------------------------------------------------------------#

#API nyckel 
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

@st.cache_data
def read_csv_file():
    # Read the CSV file into a DataFrame
    subset = pd.read_csv('subset.csv')
    return subset

# Load data using @st.cache
subset = read_csv_file()

print("Almost done!")

#--------------------------------------------------------------------------------------------------------------------------#
#CLUSTERING


#nltk.download('stopwords')
#nltk.download('punkt')

# Ladda in nltk:s stemmingfunktion för svenska
stemmer_sv = SnowballStemmer("swedish")
# Ladda in stoppord för svenska från nltk
stop_words_sv = set(stopwords.words('swedish'))
# Ladda in engelska stoppord för att hantera engelska texter
stop_words_en = set(stopwords.words('english'))
# Ladda in punktuation från string
punctuation = set(string.punctuation)

# Select only relevant columns and make a copy to avoid SettingWithCopyWarning
new_subset = subset[[
    'headline',
    'description.text'
]].copy()

print("reading nltk")


#En funktion för att förbereada...
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
new_subset['combined_text'] = new_subset.apply(lambda row: preprocess_text((row['headline'] if pd.notnull(row['headline']) else '') + ' ' + (row['description.text'] if pd.notnull(row['description.text']) else ''), language='swedish'), axis=1)


# Justera vektoriseringen för att använda olika parametrar
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(new_subset['combined_text'])

# Optimera antalet kluster med elbow method eller silhouette score

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

print("clustering done!")

 
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

# Visa titel 
st.subheader('Lediga jobb')

search_query = st.text_input('Sök efter specifika ord:', value="", help="Jobbtitel, nyckelord eller företag etc",)

region, form, time, branch = st.columns(4)


with region:
   selected_place = st.selectbox(f'Välj region:', places_list)

with form:
   selected_time_of_work = st.selectbox(f'Välj anställningsform:', time_of_work)

with time:
   selected_duration_time = st.selectbox(f'Välj tidsomfattning', duration_time)

with branch:
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

if selected_industry == 'Visa alla':
    industry_condition = subset['industry'].notna()
else:
    industry_condition = subset['industry'] == selected_industry

# Filtered subset based on all conditions
filtered_subset = subset[(region_condition) & (time_of_work_condition) & (duration_condition) & (text_condition) & (industry_condition)]
filtered_subset = filtered_subset.rename(columns=column_aliases) 

job_count = filtered_subset.shape[0]

#--------------------------------------------------------------------------------------------------------------------------#

#Visar hur många lediga jobba som finns
st.markdown(f"<h1 style='font-weight: bold; color: #4a90e2'>{job_count} st </h1>", unsafe_allow_html=True)
st.markdown("Jobb som matchar sökningen:")


# Välj endast dessa tre kolumner
ny_subset = filtered_subset[[
    'headline',
    'employer.workplace',  
    'description.text'
]]

# Titel och text högst upp
st.subheader('Lediga jobb')

print("starting with AI answers")

#antalet jobb
number = 2 
temp = st.empty()

#resultaten som visas
with temp.container():
    print("Laddar gpt")
    for i in range(min(len(ny_subset), number)):
        print(f'#{i}')
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


#--------------------------------------------------------------------------------------------------------------------------#

#SUPERVISED LEARNING

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

st.markdown("---")
st.subheader("AI-generator", help="Detta är endast en prototyp och inte en färdigt utvecklad modell")
info = """Nedan listar en AI de tre bäst lämpade arbeten för elitidrottare. 
Dessa förslag har utvecklats utifrån en supervised model 
som tränats för att ge bästa möjliga rekommendation."""

st.write(info)
st.markdown('<br>', unsafe_allow_html=True)
st.markdown("<h6 style='text-align:left;'>Top tre:</h6>", unsafe_allow_html=True)


top_predictions = sorted_df[['headline','description.text', 'prediction']].head(3)


#Gpt genererade förslag utifrån filter
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

#Panelen längst ner
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)


#Tjock linje innan
st.markdown(
    """
    <style>
        .line {
            width: 100%;
            height: 2px;
            background-color: black; /* Navy färg */
            margin-bottom: 20px;
        }
    </style>
    <div class="line"></div>
    """,
    unsafe_allow_html=True
)

#Info längst ner i kolumner
safety, ass, terms, sportee = st.columns(4)

with safety:
    st.markdown("<h6 style='text-align:left;'>Säkerhet</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Kundsäkerhet</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Hantering av kunduppgifter</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Falska mail</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Anmäl ett fel</h6>", unsafe_allow_html=True)
    

with ass:
    st.markdown("<h6 style='text-align:left;'>För föreingen</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Lägg till egen annons</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Ändra layout</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Visa alla jobb</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Inloggning för förenigar</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Administrera föreningsannonser</h6>", unsafe_allow_html=True)


with terms:
    st.markdown("<h6 style='text-align:left;'>Villkor</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Användarvillkor</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Personuppgiftshantering</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Cookies</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Cookiesinställningar</h6>", unsafe_allow_html=True)


with sportee:
    st.markdown("<h6 style='text-align:left;'>SPORTEE</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Om SPORTEE</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Press</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Jobba på SPORTEE</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Kontakta oss</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Inspiration</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Tips och guider</h6>", unsafe_allow_html=True)

