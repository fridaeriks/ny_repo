#welcome strem
import streamlit as st
import subprocess

st.markdown(
    """
    <h3 style='text-align: center;'>Välkommen till PODIUM</h3>
    """,
    unsafe_allow_html=True
)

# Funktion för att köra den andra Streamlit-filen
def run_main_app():
    subprocess.Popen(["streamlit", "run", "ny_strem.py"])


st.write("")  # Lägger till lite extra utrymme
col1, _, col2, _ = st.columns([1, 1, 6, 1])
with col2:
    if st.button("Starta podium"):
        run_main_app()