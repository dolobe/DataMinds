import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
import numpy as np
from fpdf import FPDF

st.set_page_config(page_title="Analyse de Données", layout="wide")

def clean_data(df):
    df = df.dropna()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype(str)
    for column in df.select_dtypes(include=['string']).columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def explore_data(df):
    st.write("Résumé des données :")
    st.write(df.describe())
    st.write("Types de données :")
    st.write(df.dtypes)

    st.subheader("Graphiques")
    # Histogramme
    if st.checkbox("Afficher un histogramme"):
        column = st.selectbox("Choisir une colonne", df.columns)
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        st.pyplot(plt)

    # Nuage de points
    if st.checkbox("Afficher un nuage de points"):
        x_col = st.selectbox("Choisir une colonne pour l'axe X", df.columns)
        y_col = st.selectbox("Choisir une colonne pour l'axe Y", df.columns)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[x_col], y=df[y_col])
        st.pyplot(plt)

    # Diagramme en boîte
    if st.checkbox("Afficher un diagramme en boîte"):
        column = st.selectbox("Choisir une colonne pour le box plot", df.columns)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])
        st.pyplot(plt)

    # Carte de chaleur (corrélations)
    if st.checkbox("Afficher la carte de chaleur"):
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            plt.figure(figsize=(12, 8))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
        else:
            st.warning("Aucune colonne numérique disponible pour calculer la corrélation.")

    # Graphique de paires
    if st.checkbox("Afficher un graphique de paires"):
        sns.pairplot(df)
        st.pyplot(plt)

    # Graphique à barres
    if st.checkbox("Afficher un graphique à barres"):
        column = st.selectbox("Choisir une colonne catégorique pour le bar chart", df.select_dtypes(include=['object']).columns)
        plt.figure(figsize=(10, 6))
        df[column].value_counts().plot(kind='bar')
        st.pyplot(plt)

    # Graphique linéaire
    if st.checkbox("Afficher un graphique linéaire"):
        date_column = st.selectbox("Choisir une colonne de date", df.select_dtypes(include=['datetime']).columns)
        value_column = st.selectbox("Choisir une colonne numérique", df.select_dtypes(include=[np.number]).columns)
        plt.figure(figsize=(10, 6))
        plt.plot(df[date_column], df[value_column])
        plt.xlabel(date_column)
        plt.ylabel(value_column)
        plt.title("Graphique Linéaire")
        st.pyplot(plt)

def train_ml_model(combined_df):
    st.write("Colonnes disponibles dans le DataFrame :", combined_df.columns.tolist())
    st.write(combined_df.dtypes)

    for column in combined_df.columns:
        combined_df[column] = pd.to_numeric(combined_df[column], errors='coerce')
    st.write("Valeurs manquantes par colonne :", combined_df.isnull().sum())
    combined_df.fillna(0, inplace=True)

    target_column = st.selectbox("Choisissez la colonne cible", combined_df.columns)
    X = combined_df.drop(target_column, axis=1)
    y = combined_df[target_column]

    model = RandomForestClassifier()
    model.fit(X, y)
    
    st.success("Modèle ML entraîné avec succès!")
    return model, X.columns.tolist()

def train_dl_model(df):
    st.subheader("Modèle de Deep Learning")

    target_col = st.selectbox("Choisissez la colonne cible pour DL", df.columns)
    features = df.drop(columns=[target_col])
    
    X_train, X_test, y_train, y_test = train_test_split(features, df[target_col], test_size=0.2, random_state=42)

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    loss, accuracy = model.evaluate(X_test, y_test)
    st.write(f"Précision du modèle DL : {accuracy:.2f}")

    return model, features.columns.tolist()

def predict_with_dl_model(model, features):
    st.subheader("Faire une Prédiction")
    
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(f"Entrer la valeur pour {feature}", value=0.0)

    input_df = pd.DataFrame(user_input, index=[0])

    prediction = model.predict(input_df)
    st.write(f"Prédiction : {prediction[0][0]:.2f}")

def save_to_pdf(content, filename="rapport.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for line in content.split('\n'):
        pdf.cell(0, 10, txt=line, ln=True)
    
    pdf.output(filename)
    st.success(f"Rapport sauvegardé en tant que {filename}")

def generate_sql_code(table_name):
    sql_code = f"SELECT * FROM {table_name};"
    st.code(sql_code, language='sql')

st.title("Application d'Analyse de Données avec Streamlit")

uploaded_files = st.file_uploader("Importer jusqu'à 20 fichiers CSV", type="csv", accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 20:
        st.warning("Vous ne pouvez importer que jusqu'à 20 fichiers.")
    else:
        dataframes = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            df_clean = clean_data(df)
            dataframes.append(df_clean)

        combined_df = pd.concat(dataframes, ignore_index=True)
        explore_data(combined_df)

        rf_model, rf_features = train_ml_model(combined_df)
        
        if rf_model is not None and rf_features is not None:
            dl_model, dl_features = train_dl_model(combined_df)
            predict_with_dl_model(dl_model, dl_features)

        # Sauvegarder en PDF
        if st.button("Sauvegarder le rapport en PDF"):
            save_to_pdf("Contenu de votre rapport d'analyse")

        # Générateur SQL
        st.subheader("Générateur de requêtes SQL")
        table_name = st.text_input("Entrez le nom de la table pour générer une requête SQL :")
        if st.button("Générer le code SQL"):
            if table_name:
                generate_sql_code(table_name)
            else:
                st.warning("Veuillez entrer un nom de table.")
