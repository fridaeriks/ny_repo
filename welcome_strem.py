#welcome strem
import streamlit as st
import subprocess


import streamlit_shadcn_ui as ui

st.markdown(
    """
    <h3 style='text-align: center;'>Välkommen till SPORTEE</h3>
    """,
    unsafe_allow_html=True
)

# Funktion för att köra den andra Streamlit-filen
def run_main_app():
    subprocess.Popen(["streamlit", "run", "mirko_ny.py"])


st.write("")  # Lägger till lite extra utrymme
col1, _, col2, _ = st.columns([1, 1, 1.5, 1])
st.markdown ('         .   ')
with col2:
    if st.button("Kom igång"):
        run_main_app()

#ui.button(text="Beautiful Button", key="styled_btn_tailwind", className="bg-orange-500 text-white")



# Center-align the button using Tailwind CSS classes
ui.button(text="Beautiful Button", key="styled_btn_tailwind", className="mx-10 block bg-orange-500 text-white")


import streamlit as st

# Set Streamlit theme
st.set_page_config(
    page_title="My Streamlit App",
    page_icon=":smiley:",
    layout="centered",  # Center-align content
    initial_sidebar_state="collapsed",  # Collapse the sidebar initially
    theme="orange"  # Set the theme color to orange
)

# Display the button
if st.button("Beautiful Button"):
    st.write("Button clicked!")

