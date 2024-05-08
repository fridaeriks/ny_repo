import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Läs in datasetet
df = pd.read_csv('subset.csv')

# Definiera träningsdata
train_data = df.head(206)




#st.write("First few rows of the dataset:")
#st.write(subset.head(100))

labels = ["JA", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
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


def label_to_numeric(label):
    if label == 'JA':
        return 1
    elif label == 'NEJ':
        return 0
    else:
        return None
    
train_data.loc[:, 'Label'] = labels
train_data.loc[:, 'Label'] = train_data['Label'].apply(label_to_numeric)

# Definiera egenskaper (X) och målvariabel (y) för träning
X_train = train_data.drop('Label', axis=1)
y_train = train_data['Label']

print(X_train.dtypes)

# Skapa en instans av modellen
model = SVC()

X_train_numeric = X_train.select_dtypes(include='number')

# Träna modellen
model.fit(X_train, y_train)

# Generera förutsägelser för resterande data
X_remaining = df.iloc[206:]  # De återstående 10 000 raderna
predictions_remaining = model.predict(X_remaining)

# Skapa en kopia av de återstående data och lägg till de förutsagda etiketterna
remaining_data = X_remaining.copy()
remaining_data['Predicted_Label'] = predictions_remaining

# Du kan sedan spara detta till en CSV-fil om du vill
remaining_data.to_csv('remaining_data_with_labels.csv', index=False)




#subset['Label'] = labels

#st.write("First rows with labels:")
#st.write(subset.head(206))

# Visa beskrivning för de första 100 raderna
"""for i in range(200):
    st.title("Row " + str(i+1))
    st.subheader("Headline:")
    st.write(subset['headline'].iloc[i])
    st.write(subset['working_hours_type.label'].iloc[i])
    st.subheader("Description:")
    st.write(subset['description.text'].iloc[i])"""




# Skapa en tom lista för att lagra etiketterna
#labels = []

#st.write("Data with labels:")
# Loopa igenom varje rad i datasetet
#for index, row in subset.head(100).iterrows():
    # Skriv ut raden
    #st.write(row)
     # Generera en unik nyckel för selectbox baserat på radens index
    #selectbox_key = f"label_selectbox_{index}"
    # Lägg till ett select-element för användaren att välja etikett
    #label = st.selectbox(f"Label for this row {index}:", options=["Ja", "Nej"], key=selectbox_key)
    # Lägg till etiketten i listan
    #labels.append(label)

# Lägg till den nya kolumnen med etiketterna i datasetet
#subset['Label'] = labels

# Skriv ut det uppdaterade datasetet
#st.write(subset)

#st.write("First few rows of the dataset:")
#st.write(subset.head(100))



