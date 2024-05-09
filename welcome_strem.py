#welcome strem
import streamlit as st
import subprocess

st.markdown(
    """
    <h3 style='text-align: center;'>Välkommen till <span style='color: #4a90e2'>SPORTEE</span></h3>
    """,
    unsafe_allow_html=True
)

st.markdown('''
<div style="text-align: center;">
SPORTEE effektiviserar samarbetet mellan idrottare och idrottsföreningar i Sverige. 
Genom praktiska lösningar och nära samarbete främjar vi både jobbmöjligheter och personlig 
utveckling för idrottare. Tillsammans skapar vi en mer inkluderande och hållbar idrottsmiljö. 
</div>
''', unsafe_allow_html=True)

st.markdown('<br>',unsafe_allow_html=True)

# Funktion för att köra den andra Streamlit-filen
def run_main_app():
    subprocess.Popen(["streamlit", "run", "ny_strem.py"])




col1, _, col2, _ = st.columns([1, 1, 1.8, 1])
with col2:
    if st.button("Kom igång"):
        run_main_app()


