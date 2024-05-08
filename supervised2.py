import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Läs in data
df = pd.read_csv('subset.csv').head(206)

# Ladda ned stoppord och lexikon för lemmatisering
nltk.download('stopwords')
nltk.download('punkt')

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
df['description.text'] = preprocess_text_column(df['description.text'])

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
X = df_with_labels['description.text']
y = df_with_labels['label']

# Dela upp data i träningsdata och testdata
# Dela upp data i träningsdata (150) och testdata (50) slumpmässigt
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150, test_size=50, random_state=42)

# Skapa TF-IDF-vektorer från text med samma vokabulär som användes för träning
X_train_vectorized = vectorizer.fit_transform(X_train)

# Använd samma vektoriseringsinstans för att transformera testdatan
X_test_vectorized = vectorizer.transform(X_test)

# Välj modell (Logistisk regression) och ange viktade klasser
# Du kan justera viktningen för att passa din specifika situation
model = LogisticRegression(max_iter=1000, class_weight={'NEJ': 1, 'JA': 10})

# Träna modellen
model.fit(X_train_vectorized, y_train)

# Förutsäg på testdata
y_pred = model.predict(X_test_vectorized)

# Utvärdera modellens prestanda
print(classification_report(y_test, y_pred))

# Förutsäg lämpligheten för varje jobbannons i ditt dataset
df['prediction'] = model.predict(vectorizer.transform(df['description.text']))

# Sortera DataFrame baserat på förutsägelserna för att få jobbannonserna i kronologisk ordning för vad som passar bäst med idrott
sorted_df = df.sort_values(by='prediction', ascending=False)

# Skriv ut de första 10 jobbannonserna
print(sorted_df[['description.text', 'prediction']].head(3))

import streamlit as st

st.subheader("First few rows that best fit an athlete:")
top_predictions = sorted_df[['description.text', 'prediction']].head(3)

# Loopa igenom varje rad och skriv ut informationen
for index, row in top_predictions.iterrows():
    st.write(f"Beskrivning: {row['description.text']}")
    st.write(f"Prediktion: {row['prediction']}")
    st.write("-------------")

