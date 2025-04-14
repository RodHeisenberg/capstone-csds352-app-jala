
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.express as px

st.set_page_config(page_title="Data Science - Análise de Documentos", layout="wide")

st.image("assets/logo.png", width=120)
st.title("Navegação Inteligente de Documentos")
st.markdown("**Sejam Bem Vindos!!!**" "Este app demonstra como usar embeddings e aprendizado de máquina para clusterizar, classificar e buscar documentos de forma inteligente.")

modelo = SentenceTransformer('all-MiniLM-L6-v2')

# Upload
docs = st.file_uploader("Faça upload de arquivos .txt", type="txt", accept_multiple_files=True)

if docs:
    textos = []
    nomes = []
    for doc in docs:
        nomes.append(doc.name)
        textos.append(doc.read().decode("utf-8"))

    if len(textos) >= 3:
        # Embeddings
        st.info("🔄 Gerando embeddings...")
        embeddings = modelo.encode(textos)

        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Redução com PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)

        df = pd.DataFrame({
            "Documento": nomes,
            "Texto": textos,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "Cluster": clusters
        })

        # Anomalias
        iso = IsolationForest(contamination=0.2, random_state=42)
        df["Anomalia"] = iso.fit_predict(embeddings)

        # Qualidade simulada
        df["Qualidade"] = np.where(df["Cluster"] == 0, "Alta", "Baixa")

        # Visualização
        st.subheader("Visualização Interativa dos Clusters")
        fig = px.scatter(df, x="x", y="y", color="Cluster", symbol="Anomalia",
                         hover_data=["Documento", "Qualidade"], width=900, height=500)
        st.plotly_chart(fig)

        # Busca Semântica
        st.subheader("🔍 Busca Semântica nos Documentos")
        consulta = st.text_input("Digite uma pergunta ou termo:")
        if consulta:
            consulta_vec = modelo.encode([consulta])
            sims = cosine_similarity(consulta_vec, embeddings)[0]
            idx = np.argmax(sims)
            st.success(f"Documento mais relevante: **{df.iloc[idx]['Documento']}**")
            st.code(df.iloc[idx]['Texto'][:1000])
    else:
        st.warning("É necessário fazer upload de pelo menos 3 arquivos para executar o KMeans.")
