# import pandas as pd

# # Read the updated CSV file into a DataFrame
# subset_updated = pd.read_csv('subset.csv')

# # Define the original column names
# original_column_names = {
#     'ID': 'id',
#     'Original ID': 'original_id',
#     'Headline': 'headline',
#     'Number of Vacancies': 'number_of_vacancies',
#     'Experience Required': 'experience_required',
#     'Driving License Required': 'driving_license_required',
#     'Detected Language': 'detected_language',
#     'Description': 'description.text',
#     'Duration Label': 'duration.label',
#     'Working Hours Type': 'working_hours_type.label',
#     'Employer Name': 'employer.name',
#     'Employer Workplace': 'employer.workplace',
#     'Workplace Municipality': 'workplace_address.municipality',
#     'Workplace Region': 'workplace_address.region',
#     'Region Code': 'workplace_address.region_code',
#     'Extracted Occupation': 'keywords.extracted.occupation'
# }

# # Rename the columns back to original names
# subset_original = subset_updated.rename(columns=original_column_names)

# # Write the DataFrame with original column names to CSV
# subset_original.to_csv('subset.csv', index=False)
import os
import streamlit as st
import streamlit_shadcn_ui as ui






st.set_page_config(
    page_title="streamlit-folium documentation: Draw Support",
    page_icon=":pencil:",
    layout="wide",
)
pages = st.sidebar.container()

"""
# streamlit-folium: Draw Support

Folium supports some of the [most popular leaflet
plugins](https://python-visualization.github.io/folium/plugins.html). In this example,
we can add the
[`Draw`](https://python-visualization.github.io/folium/plugins.html#folium.plugins.Draw)
plugin to our map, which allows for drawing geometric shapes on the map.

When a shape is drawn on the map, the coordinates that represent that shape are passed
back as a geojson feature via the `all_drawings` and `last_active_drawing` data fields.

Draw something below to see the return value back to Streamlit!
"""

with st.echo(code_location="below"):
    import folium
    import streamlit as st
    from folium.plugins import Draw

    from streamlit_folium import st_folium

    m = folium.Map(location=[39.949610, -75.150282], zoom_start=5)
    Draw(export=True).add_to(m)

    c1, c2 = st.columns(2)
    with c1:
        output = st_folium(m, width=700, height=500)

    with c2:
        st.write(output)




# Define the hover card inside the main content area
ui.hover_card(label="Hover on me1!", content="I am a hover card1!", content_type="text", key="hover_card_1")


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

