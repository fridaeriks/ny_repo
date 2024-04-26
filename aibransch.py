import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import streamlit as st

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

# Select only relevant columns and make a copy to avoid SettingWithCopyWarning
subset = jobtech_dataset[[
    'id',
    'headline',
    'keywords.extracted.occupation'
]].copy()

# Define a function to preprocess text
def preprocess_text(text):
    # Convert list of keywords to a single string and then convert to lowercase
    if isinstance(text, list):
        text = ' '.join(text)
    text = text.lower()
    return text

# Apply text preprocessing
subset['headline'] = subset['headline'].apply(preprocess_text)
subset['keywords.extracted.occupation'] = subset['keywords.extracted.occupation'].apply(preprocess_text)

# Combine headline and keywords into a single text for each job ad
subset['combined_text'] = subset['headline'] + ' ' + subset['keywords.extracted.occupation']

# Justera vektoriseringen för att använda olika parametrar
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(subset['combined_text'])

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

# Streamlit app
st.title('Jobtech Dataset Clustering')
# Display the DataFrame with cluster labels and industry names
st.write(subset)
