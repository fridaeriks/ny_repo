import requests
import zipfile
import io
import json
from openai import OpenAI
import streamlit as st
import os



# Execute the code snippet
print("Running")

if not os.path.exists("dataset.jsonl"):
    url = "https://data.jobtechdev.se/annonser/historiska/2023.jsonl.zip"

    try:
        zip = requests.get(url)
        zip.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(zip.content)) as zip_file:
            file_name = zip_file.namelist()[0]
            jsonl_file = zip_file.open(file_name)
            print("File loaded successfully!")

            # Write to the JSONL file
            with open("dataset.jsonl", "ab") as output_file:  # Use "ab" mode to append
                for line in jsonl_file:
                    output_file.write(line)
            print("JSONL data appended successfully!")

    except requests.exceptions.RequestException as e:
        print("Failed to fetch data:", e)
    except Exception as e:
        print("Error:", e)

API_KEY = open('Open_AI_key', 'r').read()

client = OpenAI(
    api_key=API_KEY
) 

def load_job_ads(file_path):
    job_ads = []
    with open("dataset.jsonl", "r", encoding='utf-8') as file:
        for line_number, line in enumerate(file):
            if line_number >= 100000:
                break
            job_ads.append(json.loads(line))
    return job_ads

job_ads = load_job_ads('2023.jsonl')

def main():
    print("Started sucessfully!")
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