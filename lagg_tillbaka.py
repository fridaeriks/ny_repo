"""#Show more options
if len(ny_subset) > number:
    if st.button('Visa fler'):
        number += 10
        for i in range(number - 10, min(len(ny_subset), number)):
            with st.expander(f"{ny_subset['Rubrik'].iloc[i]}"):
                st.write(f"Arbetsgivare: {ny_subset['Arbetsgivare'].iloc[i]}")
                st.write(f"Arbetsbeskrivning: {ny_subset['Beskrivning'].iloc[i]}")"""
#plats 210