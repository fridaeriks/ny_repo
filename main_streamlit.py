import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json('dataset.jsonl', lines=True)

# Visa tabellen i Streamlit
st.write(df)

df = pd.read_json('subset')

subset_df = df[['description', 'working_hours_type']]
subset_df = subset_df.rename(columns={'description': 'Annons beskrivning', 'working_hours_type': 'Tid'})

st.write("<span style='font-size: 20px;'>Här ser du de deltidsjobb som vi kan erbjuda:</span>", unsafe_allow_html=True)


df = pd.read_json('dataset.jsonl')
st.dataframe(subset_df)