
import pandas as pd
import json
import streamlit as st

# Load the JSON file into a DataFrame 
lines = []
with open('dataset.jsonl', 'rb') as file: 
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

#Titel och text högst upp
st.title('Vårt namn')
st.markdown("Info om vårt projekt")
st.markdown("---")
st.write(subset)



#Den gråa sidopanelen
left_column = st.sidebar.empty()

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

left_column.markdown("""
<div class="left-column" style="padding: 20px;">
<h3>Filter</h3>
<p>Här kan man lägga till text om man vill.</p>
<hr>
<p style="font-size: 18px;">Våra kontaktuppgifter:</p> 
<p style="font-size: 12px;">Vera@devil.com</p> 
                     
</div>
""", unsafe_allow_html=True)


# Display description of a specific row
row_index = st.slider("Select Row Index", 0, len(subset)-1, 25)
st.subheader("Description for Selected Row:")
st.write(subset['description.text'].iloc[row_index])

# Show the variables in the dataset (equivalent to column names)
st.write("Columns in the dataset:")
st.write(subset.columns)
st.write('')







#Tabell där man kan filtrera med båda rullistorna
column_aliases = {
    'headline': 'Rubrik',
    'number_of_vacancies': 'Antal Lediga Platser',
    'description.text': 'Beskrivning',
    'working_hours_type.label': 'Tidsomfattning',
    'workplace_address.region': 'Region',
    'workplace_address.municipality': 'Kommun'}


places_list = subset['workplace_address.region'].dropna().unique().tolist()
if 'None' in places_list:
    places_list.remove('None')


time_of_work = subset['working_hours_type.label'].dropna().unique().tolist()
if 'None' in time_of_work:
    time_of_work.remove('None')

selected_place = st.selectbox("Select Region:", places_list)
selected_time_of_work = st.selectbox("Select Time of Work:", time_of_work)

filtered_subset = subset[(subset['workplace_address.region'] == selected_place) & 
                         (subset['working_hours_type.label'] == selected_time_of_work)]

filtered_subset = filtered_subset[['headline', 'number_of_vacancies', 'description.text', 
                                   'working_hours_type.label', 'workplace_address.region', 
                                   'workplace_address.municipality']]

filtered_subset = filtered_subset.rename(columns=column_aliases) 
st.write(filtered_subset)