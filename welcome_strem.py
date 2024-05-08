#welcome strem
import streamlit as st
import subprocess

st.markdown(
    """
    <h3 style='text-align: center;'>Välkommen till SPORTEE</h3>
    """,
    unsafe_allow_html=True
)

# Funktion för att köra den andra Streamlit-filen
def run_main_app():
    subprocess.Popen(["streamlit", "run", "ny_strem.py"])



st.markdown(
    """
    <style>
        .centered-button {
            display: block;
            margin: auto;
            background-color: #0072c6; /* Blå färg */
            color: white; /* Vit textfärg */
            border: none; /* Ingen kantlinje */
            padding: 10px 20px; /* Padding för knappens storlek */
            border-radius: 5px; /* Runda kanter */
            cursor: pointer; /* Visa pekare vid hover */
            font-size: 16px; /* Storlek på text */
        }
        .centered-button:hover {
            background-color: #005ea3; /* Ändra färg vid hover */
        }
    </style>
    """,
    unsafe_allow_html=True
)

col1, _, col2, _ = st.columns([1, 1, 1.5, 1])
with col2:
    if st.button("Kom igång"):
        run_main_app()


