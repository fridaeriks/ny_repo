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

st.write("")  # Lägger till lite extra utrymme
col1, _, col2, _ = st.columns([1, 1, 1.5, 1])
st.markdown ('         .   ')

with col2:
    if st.button("Kom igång"):
        run_main_app()





# Center-align the button using Tailwind CSS classes
ui.button(text="Beautiful Button", key="styled_btn_tailwind", className="mx-10 block bg-orange-500 text-white")




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


