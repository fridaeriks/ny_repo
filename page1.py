import pandas as pd
import json
import streamlit as st
from openai import OpenAI
import os
import requests
import io
import zipfile
import numpy as np


st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))


x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option


# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")
    



# Load data from CSV file
df = pd.read_csv("subset.csv")

# Search bar
search_query = st.text_input("Search by Name", "")

# Filtering data based on search query
if search_query:
    filtered_df = df[df['Name'].str.contains(search_query, case=False)]
else:
    filtered_df = df

# Display filtered data
st.write(filtered_df)