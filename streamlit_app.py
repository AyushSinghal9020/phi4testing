import streamlit as st 
import requests 

audio_input = st.audio_input('Speak')

if st.button('Ask') : 

    with open('audio.wav' , 'wb') as audio_file : audio_file.write(audio_input.getbuffer())

    response = requests.get('http://localhost:8000/ask' , params = {'audio_file' : 'audio.wav'})

    st.write(response.json())