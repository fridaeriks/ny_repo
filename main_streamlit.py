import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.markdown('**<span style="color: black; font-size: 34px;">VÅRT NAMN</span>**', unsafe_allow_html=True)
st.markdown('<span style="color: grey; font-size: 20px;" >Kort text som beskriver vilka vi är/gör</span>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True) 
st.markdown('<span style="color: grey; font-size: 18px;" >Målet är... </span>', unsafe_allow_html=True)

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
<p>Här kan du filtrera på stad, branch, tid etc.</p>
<hr>
<p style="font-size: 18px;">Våra kontaktuppgifter:</p> 
<p style="font-size: 12px;">Vera@devil.com</p> 
                     
</div>
""", unsafe_allow_html=True)

right_column = st.sidebar

df = pd.read_json('subset')

subset_df = df[['description', 'working_hours_type']]
subset_df = subset_df.rename(columns={'description': 'Annons beskrivning', 'working_hours_type': 'Tid'})

st.write("<span style='font-size: 20px;'>Här ser du de deltidsjobb som vi kan erbjuda:</span>", unsafe_allow_html=True)


df = pd.read_json('dataset.jsonl')
st.dataframe(subset_df)