import streamlit as st
import pandas as pd


st.title('Tuto :red[Streamlit]')
st.write('Hello Master IFRI')
st.markdown("# Dataset visualizer")
st.warning("#### Vous devez uploader un fichier qui sera analysé. ")

checbox = st.checkbox('Cliquez voir ')

dataset_file = st.file_uploader("Téléverser le fichier csv", type=['csv'])

if dataset_file:
    st.success("Super ! Vous avez bel et bien téléversé le fichier.")
    
    btn = st.button(':green[**Analyser**]')
    file = pd.read_csv(dataset_file)
    st.write(file.info())
    
