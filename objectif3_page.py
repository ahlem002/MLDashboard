import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES_OBJ3 = [
    "Age",
    "Current Employee Rating",
    "Engagement Score",
    "Satisfaction Score",
    "Work-Life Balance Score",
    "Training Duration(Days)",
    "Training Cost",
]


@st.cache_data
def run_objectif3_clustering(df_hr: pd.DataFrame, final_k: int = 2):
    used_cols = [c for c in FEATURES_OBJ3 if c in df_hr.columns]

    data = df_hr[used_cols].apply(pd.to_numeric, errors="coerce").dropna().copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    scores = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append({"Model": "KMeans", "K": k, "Silhouette": score})

    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    agg = AgglomerativeClustering(n_clusters=final_k)
    agg_labels = agg.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    result_df = data.copy()
    result_df["KMeans_Cluster"] = kmeans_labels
    result_df["Agglomerative_Cluster"] = agg_labels

    profile = result_df.groupby("KMeans_Cluster").mean(numeric_only=True).round(2)

    return {
        "used_cols": used_cols,
        "scores": pd.DataFrame(scores),
        "result_df": result_df,
        "profile": profile,
        "X_pca": X_pca,
        "explained": pca.explained_variance_ratio_.sum(),
        "kmeans_score": silhouette_score(X_scaled, kmeans_labels),
        "agg_score": silhouette_score(X_scaled, agg_labels),
    }

def render_objectif3_page(df_hr: pd.DataFrame):
    st.header("Objectif 3 — Segmentation des collaborateurs")

    st.write(
        "Cette page applique un apprentissage non supervisé pour regrouper "
        "les collaborateurs selon des caractéristiques RH numériques."
    )

    final_k = st.radio(
        "Nombre de clusters final",
        [2, 3],
        index=0,
        horizontal=True,
        help="K=2 est plus stable pour ce dataset. K=3 est disponible pour comparaison."
    )

    result = run_objectif3_clustering(df_hr, final_k=final_k)

    c1, c2, c3 = st.columns(3)
    c1.metric("Employés utilisés", f"{len(result['result_df']):,}")
    c2.metric("Features utilisées", f"{len(result['used_cols'])}")
    c3.metric("Variance PCA 2D", f"{result['explained']:.2%}")

    st.subheader("Features utilisées")
    st.write(", ".join(result["used_cols"]))

    st.subheader("Comparaison des valeurs de K")
    st.dataframe(result["scores"].round(4), width="stretch", hide_index=True)

    fig_scores = px.line(
        result["scores"],
        x="K",
        y="Silhouette",
        markers=True,
        title="Silhouette Score selon le nombre de clusters",
    )
    st.plotly_chart(fig_scores, width="stretch")

    st.subheader("Résultat K-Means + PCA")

    pca_df = pd.DataFrame({
        "PCA Component 1": result["X_pca"][:, 0],
        "PCA Component 2": result["X_pca"][:, 1],
        "Cluster": result["result_df"]["KMeans_Cluster"].astype(str),
    })

    fig = px.scatter(
        pca_df,
        x="PCA Component 1",
        y="PCA Component 2",
        color="Cluster",
        title="Segmentation des collaborateurs avec K-Means et PCA",
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=7))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Scores finaux")
    s1, s2 = st.columns(2)
    s1.metric("K-Means Silhouette", f"{result['kmeans_score']:.4f}")
    s2.metric("Agglomerative Silhouette", f"{result['agg_score']:.4f}")

    if result["kmeans_score"] < 0.2:
        st.warning(
            "Le score de silhouette est faible. Cela indique que les groupes ne sont pas fortement séparés. "
            "C’est normal dans des données RH, car les profils des employés sont souvent proches et continus."
        )
    else:
        st.success(
            "Le score de silhouette montre une séparation acceptable entre les groupes."
        )

    st.subheader("Profil moyen par cluster")
    profile = result["profile"].copy()
    profile.index = [f"Cluster {idx}" for idx in profile.index]
    st.dataframe(profile, width="stretch")

    st.info(
        "Conclusion: cette segmentation est exploratoire. "
        "Elle aide à identifier des groupes généraux de collaborateurs, mais elle ne doit pas être utilisée seule "
        "pour prendre des décisions RH importantes."
    )
    st.header("Objectif 3 — Segmentation des collaborateurs")

    st.write(
        "Cette page applique un apprentissage non supervisé pour regrouper "
        "les collaborateurs selon des caractéristiques RH numériques."
    )

    final_k = st.radio(
        "Nombre de clusters final",
        [2, 3],
        index=0,
        horizontal=True,
    )

    result = run_objectif3_clustering(df_hr, final_k=final_k)

    c1, c2, c3 = st.columns(3)
    c1.metric("Employés utilisés", f"{len(result['result_df']):,}")
    c2.metric("Features utilisées", f"{len(result['used_cols'])}")
    c3.metric("Variance PCA 2D", f"{result['explained']:.2%}")

    st.subheader("Features utilisées")
    st.write(result["used_cols"])

    st.subheader("Comparaison des valeurs de K")
    st.dataframe(result["scores"].round(4), width="stretch", hide_index=True)

    fig_scores = px.line(
        result["scores"],
        x="K",
        y="Silhouette",
        markers=True,
        title="Silhouette Score selon K",
    )
    st.plotly_chart(fig_scores, width="stretch")

    st.subheader("Résultat K-Means + PCA")

    pca_df = pd.DataFrame({
        "PCA Component 1": result["X_pca"][:, 0],
        "PCA Component 2": result["X_pca"][:, 1],
        "Cluster": result["result_df"]["KMeans_Cluster"].astype(str),
    })

    fig = px.scatter(
        pca_df,
        x="PCA Component 1",
        y="PCA Component 2",
        color="Cluster",
        title="Segmentation des collaborateurs avec K-Means et PCA",
        opacity=0.75,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Scores finaux")
    s1, s2 = st.columns(2)
    s1.metric("K-Means Silhouette", f"{result['kmeans_score']:.4f}")
    s2.metric("Agglomerative Silhouette", f"{result['agg_score']:.4f}")

    st.subheader("Profil moyen par cluster")
    st.dataframe(result["profile"], width="stretch")

    st.info(
        "Interprétation: le clustering est exploratoire. "
        "Si les clusters se chevauchent, cela signifie que les profils RH sont proches "
        "et pas parfaitement séparables."
    )