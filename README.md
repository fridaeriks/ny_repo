# Projekt verksamhetsstyrning

## Instruktioner
### Installera moduler:
    pip install -r requirements.txt

### Lägg till OpenAI API Key
1. Skapa en fil som heter OpenAI_API_key
2. Se till att OpenAI_API_key är med i .gitignore!!!
3. klistra in din api-key i den nya filen du skapat.



##VIKTIGA KOLUMNER I 2023.jsonl
1. id
2. external_id
3. original_id
4. headline
5. number_of_vacancies
6. experience_required
7. driving_license_required
8. detected_language
9. description.text
10. description.conditions
11. salary_type.label
12. duration.label
13. working_hours_type.label
14. employer.name 
15. employer.workplace
16. workplace_adress.municipality
17. workplace_adress.municipality.code
18. workplace_adress.region
19. workplace_adress.region_code
20. keywords.extracted.occupation