
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris, fetch_california_housing
import numpy as np

def load_dataset(name):
    if name == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif name == "california_housing":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif name == "diamonds":
        return sns.load_dataset("diamonds")
    elif name == "penguins":
        return sns.load_dataset("penguins")

def main():
    st.title("Visualisation de données avec Streamlit \n ## By Stanislas Fructueux HOUETO")
    
    # Sidebar pour les contrôles
    st.sidebar.header("Paramètres")
    
    # Sélection du dataset
    dataset_name = st.sidebar.selectbox(
        "Choisir un dataset:",
        ("iris", "california_housing", "diamonds", "penguins")
    )
    
    # Chargement des données
    df = load_dataset(dataset_name)
    
    # Affichage du nombre de lignes
    n_rows = st.sidebar.slider("Nombre de lignes à afficher:", 1, 50, 10)
    
    # Type de graphique
    plot_type = st.sidebar.selectbox(
        "Type de graphique:",
        ("Nuage de points", "Boîte à moustaches", "Histogramme")
    )
    
    # Sélection des variables
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if plot_type in ["Nuage de points", "Boîte à moustaches"]:
        x_col = st.sidebar.selectbox("Variable X:", df.columns)
        y_col = st.sidebar.selectbox("Variable Y:", numeric_columns)
        color_col = st.sidebar.selectbox("Variable de couleur (optionnel):", 
                                       ["Aucune"] + list(df.columns))
    else:  # Histogramme
        x_col = st.sidebar.selectbox("Variable:", numeric_columns)
        y_col = None
        color_col = st.sidebar.selectbox("Variable de couleur (optionnel):", 
                                       ["Aucune"] + list(df.columns))
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["📊 Graphique", "📑 Résumé", "🗃 Données"])
    
    with tab1:
        st.subheader("Visualisation")
        
        if plot_type == "Nuage de points":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=None if color_col == "Aucune" else color_col,
                title=f"Nuage de points: {x_col} vs {y_col}"
            )
            st.plotly_chart(fig)
            
        elif plot_type == "Boîte à moustaches":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=None if color_col == "Aucune" else color_col,
                title=f"Boîte à moustaches: {y_col} par {x_col}"
            )
            st.plotly_chart(fig)
            
        else:  # Histogramme
            fig = px.histogram(
                df,
                x=x_col,
                color=None if color_col == "Aucune" else color_col,
                title=f"Histogramme de {x_col}"
            )
            st.plotly_chart(fig)
    
    with tab2:
        st.subheader("Résumé statistique")
        st.write(df.describe())
        
        if st.checkbox("Afficher les informations sur les colonnes"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
    
    with tab3:
        st.subheader("Données brutes")
        st.write(df.head(n_rows))
        
        if st.checkbox("Télécharger les données"):
            st.download_button(
                label="Télécharger en CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f'{dataset_name}.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()





# import streamlit as st
# import pandas as pd
#
#
# st.title('Tuto :red[Streamlit]')
# st.write('Hello Master IFRI')
# st.markdown("# Dataset visualizer")
# st.warning("#### Vous devez uploader un fichier qui sera analysé. ")
#
# checbox = st.checkbox('Cliquez voir ')
#
# dataset_file = st.file_uploader("Téléverser le fichier csv", type=['csv'])
#
# if dataset_file:
#     st.success("Super ! Vous avez bel et bien téléversé le fichier.")
#     
#     btn = st.button(':green[**Analyser**]')
#     file = pd.read_csv(dataset_file)
#     st.write(file.info())
#     
