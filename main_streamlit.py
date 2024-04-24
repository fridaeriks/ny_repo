import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json('dataset.jsonl', lines=True)

# Visa tabellen i Streamlit
st.write(df)
