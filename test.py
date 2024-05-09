import pandas as pd
import json
import streamlit as st
import openai
from openai import OpenAI
import os
import requests
import io
import zipfile

st.image('logo2.jpg', width=300)  



# Centered image using CSS within HTML
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="logo2.jpg" alt="Logo" width="260">
    </div>
    """,
    unsafe_allow_html=True
)
