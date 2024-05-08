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




col1, _, col2, _ = st.columns([1, 1, 1.8, 1])
with col2:
    if st.button("Kom igång"):
        run_main_app()


