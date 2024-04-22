import json
from openai import OpenAI
import streamlit as st

API_KEY = open('Open_AI_key', 'r').read()

client = OpenAI(
    api_key=API_KEY
) 

# Läs innehållet från "2023.jsonl" filen

def load_job_ads(file_path):
    job_ads = []
    with open('2023.jsonl', "r", encoding='utf-8') as file:
        for line_number, line in enumerate(file):
            if line_number >= 100000:
                break
            job_ads.append(json.loads(line))
    return job_ads

job_ads = load_job_ads('2023.jsonl')

def main():
    st.title("Deltid Jobbannonser")  

    if st.button("Visa jobbannonser som är deltidsarbete"):

        response = client.chat.completions.create(
           
            messages=[
                {
                    'role': 'system', 
                    'content': 'Du är ett program som sitter på massa information om yrken. Du kan endast svara i en rullista.'
                 },
                {
                    'role': 'user', 
                    'content': 'Kan du kategorisera olika brancher som finns utifrån yrken som tillåter deltidsarbete?'
                 }

            ],
            model='gpt-3.5-turbo'
        )

        print((response.choices[0].message.content).split('\n'))

        st.write("Svar från OpenAI:")
        try:
            response_text = (response.choices[0].message.content).split('\n')
            st.write(response_text)
            
        except Exception as e:
            st.write("Kunde inte hämta svar från OpenAI:", e)

if __name__ == "__main__":
    main()

#st.write(response.choices[0].text.strip())
#print(response)
#response_text = response.choices[0].get("text", "").strip()
#st.write(response_text)