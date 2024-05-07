# """
# option = st.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone"))
# st.write("You selected:", option)
# """



df = pd.read_csv("subset.csv")

places_list = df['workplace_address.region'].dropna().unique().tolist()
places_list.insert(0, 'Visa alla')


time_of_work = df['working_hours_type.label'].dropna().unique().tolist()
time_of_work.insert(0, 'Visa alla')

duration_time = df['duration.label'].dropna().unique().tolist()
duration_time.insert(0, 'Visa alla')

#Försök att få in att sökfunktionen ska funka med det andra
keywords_list = df['keywords.extracted.occupation'].dropna().unique().tolist()
keywords_list.insert(0, 'Sök')


def keywords(df, query):
    return df[df['description.text'].str.contains(query, case=False, na=False)]

# Filter DataFrame based on search query
search_key = keywords(df, search_query)
selected_columns = ['employer.workplace', 'description.text', 'keywords.extracted.occupation' ]

# Display filtered DataFrame - Denna ska bort sen!
st.write(search_key[selected_columns])


# Display the DataFrame
st.subheader('Sök bland lediga jobb')

selected_place = st.selectbox(f'Välj region:', places_list)
selected_time_of_work = st.selectbox(f'Välj anställningsform:', time_of_work)
selected_duration_time = st.selectbox(f'Välj tidsomfattning', duration_time)
search_query = st.text_input('Search by word:')

# En funktion för att visa sökfunktionen

def keywords(df, query):
    return df[df['keywords.extracted.occupation'].str.contains(query, case=False, na=False)]

# Filter DataFrame based on search query
search_key = keywords(df, search_query)
selected_columns = ['employer.workplace', 'keywords.extracted.occupation' ]

# Display filtered DataFrame - Denna ska bort sen!
st.write(search_key[selected_columns])



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



filtered_subset = df[(region_condition) & (time_of_work_condition) & (duration_condition)]

ny_subset = filtered_subset[['headline', 'employer.workplace',
            'number_of_vacancies', 'keywords.extracted.occupation' 
                             ]]

#filtered_subset = filtered_subset.rename(columns=column_aliases) 

#HÄR HAR JAG ÄNDRAT

job_count = filtered_subset.shape[0]

st.markdown(f"<h1 style='font-weight: bold; color: green;'>{job_count} st </h1>", unsafe_allow_html=True)
st.markdown("Jobb som matchar sökningen:")


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

