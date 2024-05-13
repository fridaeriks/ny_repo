# Define a function to simplify descriptions using OpenAI API and cache the results
@st.cache
def simplify_descriptions(descriptions):
    simplified_descriptions = []
    for description in descriptions:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Du är expert på att skriva snygga jobbannonser"},
                {"role": "user", "content": description},
            ]
        )
        simplified_description = response.choices[0].message.content
        simplified_descriptions.append(simplified_description)
    return simplified_descriptions

# Batch descriptions and simplify them with optimized API calls
def process_descriptions(descriptions, batch_size=5):
    simplified_descriptions = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        simplified_batch = simplify_descriptions(batch)
        simplified_descriptions.extend(simplified_batch)
    return simplified_descriptions

# Number of job advertisements to display initially
number = 5 
temp = st.empty()

# Initial loading of simplified descriptions
simplified_descriptions = process_descriptions(ny_subset['description.text'].iloc[:number])

# Display the results using simplified descriptions
with temp.container():
    for i in range(number):
        with st.expander(f"Jobbannons {i+1} - {ny_subset['headline'].iloc[i]}"):
            st.write("-------------------------------------------------")
            st.write(simplified_descriptions[i])

# Load more job advertisements
if len(ny_subset) > number:
    if st.button('Visa fler'):
        temp.empty()
        number += number
        temp = st.empty()
        with temp.container():
            simplified_descriptions = process_descriptions(ny_subset['description.text'].iloc[:number])
            for i in range(number - number, min(len(ny_subset), number)):
                with st.expander(f"Jobbannons {i+1} - {ny_subset['headline'].iloc[i]}"):
                    st.write("-------------------------------------------------")
                    st.write(simplified_descriptions[i])
