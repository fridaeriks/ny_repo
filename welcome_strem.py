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
utveckling för idrottare.
</div>
''', unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

st.markdown('''
<div style="text-align: center;">
<span style="color: #4a90e2;">Tillsammans skapar vi en mer inkluderande och hållbar idrottsmiljö.</span>
</div>
''', unsafe_allow_html=True)


st.markdown('<br>',unsafe_allow_html=True)

# Funktion för att köra den andra Streamlit-filen
def run_main_app():
    subprocess.Popen(["streamlit", "run", "ny_strem.py"])


st.markdown("<br>", unsafe_allow_html=True)
st.markdown('''
<div style="text-align: center; font-size: 18px;">
Vad vill du göra? 
</div>
''', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


col1, col2, col3 = st.columns([1.1, 0.8, 1.8])

# Visa den befintliga knappen i den tredje kolumnen
with col3:
    if st.button("Söka jobb"):
        run_main_app()

# Skapa en tom plats bredvid den befintliga knappen i den andra kolumnen
with col2:
    if st.button("Lägga till jobb"):
        pass

# Skapa en tom plats bredvid den befintliga knappen i den första kolumnen
with col1:
    st.empty()

