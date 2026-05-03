import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


st.set_page_config(layout="wide", page_title="Dashboard ML - Objectifs")

st.markdown(
    """
    <style>
      :root { color-scheme: light; }
            .stApp { background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%); color: #102a43; }
      section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid #d6e4ff;
      }
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] span,
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3,
      section[data-testid="stSidebar"] small,
      section[data-testid="stSidebar"] .stMarkdown,
      section[data-testid="stSidebar"] [data-testid="stCaption"] {
                color: #13315c !important;
      }

            section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] {
                color: #13315c !important;
                background: #f4f8ff;
                border: 1px solid #d6e4ff;
                border-radius: 12px;
                margin: 0 0 8px 0;
                padding: 8px 10px;
                transition: all 0.15s ease;
      }

            /* Hide the default radio circle and keep a clean "tab" look. */
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
                display: none;
      }

            section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] p {
                color: #13315c !important;
                font-weight: 500;
      }

            section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
                background: #e7f0ff;
                border-color: #2f6fed;
                box-shadow: inset 0 0 0 1px #2f6fed, 0 10px 24px rgba(47, 111, 237, 0.2);
            }

            section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
                background: #edf4ff;
                border-color: #8aaef8;
            }

            .stAlert,
            div[data-testid="stMetric"],
            div[data-testid="stDataFrame"],
            div[data-testid="stTable"] {
                background: #ffffff;
                border: 1px solid #dbe8ff;
                border-radius: 14px;
                box-shadow: 0 8px 24px rgba(33, 88, 185, 0.08);
            }

            div[data-testid="stMetric"] * {
                color: #102a43 !important;
            }

            .block-container {
                padding-top: 1.4rem;
            }

            h1, h2, h3 {
                color: #0b3954;
            }

            p, li, label, span, div {
                color: #102a43;
            }
    </style>
    """,
    unsafe_allow_html=True,
)

DATASET_JOBS = "JobsDatasetProcessed.csv"
DATASET_HR = "Cleaned_HR_Data_Analysis.csv"
JOB_TITLE_COL = "Job Title"
PROJECT_ROOT = Path(__file__).resolve().parent
SKIP_SCAN_NAMES = {"app.py"}
SKIP_DIR_NAMES = {".venv", "venv", "__pycache__", ".git", "node_modules"}


@st.cache_data
def load_csv(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _read_notebook_sources(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    parts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            parts.append("".join(src))
        else:
            parts.append(str(src))
    return "\n".join(parts)


def collect_project_code_text(root: Path) -> tuple[str, list[str]]:
    """Aggregate .py / .ipynb sources for signal detection (excludes dashboard + venv)."""
    chunks: list[str] = []
    rel_files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            if name in SKIP_SCAN_NAMES:
                continue
            p = Path(dirpath) / name
            rel = str(p.relative_to(root))
            try:
                if name.endswith(".py"):
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    chunks.append(text)
                    rel_files.append(rel)
                elif name.endswith(".ipynb"):
                    text = _read_notebook_sources(p)
                    if text.strip():
                        chunks.append(text)
                        rel_files.append(rel)
            except OSError:
                continue
    return "\n\n".join(chunks), rel_files


@st.cache_data
def scan_ml_code_signals(_root_key: str) -> dict:
    """Infer which ML blocks exist in user notebooks/scripts (not app.py)."""
    root = Path(_root_key)
    blob, files = collect_project_code_text(root)
    low = blob.lower()
    return {
        "files_scanned": files,
        "skills_extraction": bool(
            re.search(
                r"it skills|soft skills|skill_frequency|extract_skills|competences|"
                r"technical_skills|tfidfvectorizer|skill",
                low,
            )
        ),
        "classification": bool(
            re.search(
                r"confusion_matrix|randomforest|logisticregression|kneighbors|"
                r"multinomialnb|accuracy_score|classification_report|train_test_split",
                low,
            )
        ),
        "clustering": bool(
            re.search(r"kmeans|agglomerativeclustering|silhouette_score|clustering", low)
        ),
        "pca": bool(re.search(r"\bpca\b|sklearn\.decomposition", low)),
    }


def parse_skill_list(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    parts = [p.strip(" \"'[]") for p in re.split(r"[,;]", text)]
    return [p for p in parts if p]


def skill_frequency(df: pd.DataFrame, col: str, top_n: int = 15):
    all_skills = []
    for val in df[col].dropna().tolist():
        all_skills.extend(parse_skill_list(val))
    if not all_skills:
        return pd.DataFrame(columns=["skill", "count"])
    out = pd.Series(all_skills).value_counts().head(top_n).reset_index()
    out.columns = ["skill", "count"]
    return out


def plot_category_bar(series: pd.Series, title: str, max_cats: int = 12):
    """Horizontal bar chart so long labels stay readable (unlike st.bar_chart)."""
    vc = series.value_counts().head(max_cats).iloc[::-1]
    df_plot = vc.reset_index()
    df_plot.columns = ["category", "count"]
    fig = px.bar(
        df_plot,
        x="count",
        y="category",
        orientation="h",
        title=title,
        height=420 + min(len(df_plot), max_cats) * 18,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f0f4f8",
        font=dict(color="#102a43", size=12),
        margin=dict(l=8, r=8, t=48, b=8),
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, width='stretch')


@st.cache_data
def run_model_comparison(df_jobs: pd.DataFrame):
    text_col = "Description" if "Description" in df_jobs.columns else "text"
    label_col = JOB_TITLE_COL if JOB_TITLE_COL in df_jobs.columns else "label"
    data = df_jobs[[text_col, label_col]].dropna().copy()
    data[text_col] = data[text_col].astype(str)
    data[label_col] = data[label_col].astype(str)

    counts = data[label_col].value_counts()
    valid_labels = counts[counts >= 2].index
    data = data[data[label_col].isin(valid_labels)]

    n_samples = len(data)
    n_classes = data[label_col].nunique()
    if n_samples < 4 or n_classes < 2:
        raise ValueError(
            "Pas assez de donnees pour comparer les modeles (au moins 2 classes et 4 lignes)."
        )

    # Keep stratified split valid for high-cardinality labels.
    # With many classes, test set needs at least one sample per class.
    min_test_size = n_classes
    target_test_size = int(np.ceil(0.2 * n_samples))
    max_test_size = n_samples - n_classes
    test_size = max(min_test_size, min(target_test_size, max_test_size))

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data[text_col],
            data[label_col],
            test_size=test_size,
            random_state=42,
            stratify=data[label_col],
        )
    except ValueError:
        # Fallback for edge cases where stratification is still infeasible.
        fallback_test_size = int(np.ceil(0.2 * n_samples))
        fallback_test_size = min(max(1, fallback_test_size), n_samples - 1)
        X_train, X_test, y_train, y_test = train_test_split(
            data[text_col],
            data[label_col],
            test_size=fallback_test_size,
            random_state=42,
            stratify=None,
        )

    vec = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
    x_train_vec = vec.fit_transform(X_train)
    x_test_vec = vec.transform(X_test)

    models = {
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=42,
        ),
    }

    rows = []
    best_name = None
    best_pred = None
    best_acc = -1.0
    for name, model in models.items():
        model.fit(x_train_vec, y_train)
        pred = model.predict(x_test_vec)
        acc = accuracy_score(y_test, pred)
        f1m = f1_score(y_test, pred, average="macro", zero_division=0)
        rows.append({"Modele": name, "Accuracy": acc, "F1 Macro": f1m})
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pred = pred

    bench = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    cm = confusion_matrix(y_test, best_pred, labels=sorted(y_test.unique()))
    return bench, best_name, cm, sorted(y_test.unique())


@st.cache_data
def run_clustering(df_hr: pd.DataFrame):
    candidates = [
        "Age",
        "Current Employee Rating",
        "Engagement Score",
        "Satisfaction Score",
        "Work-Life Balance Score",
        "Training Duration(Days)",
        "Training Cost",
    ]
    num_cols = [c for c in candidates if c in df_hr.columns]
    X = df_hr[num_cols].apply(pd.to_numeric, errors="coerce").dropna().copy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    records = []
    best_score = -1.0
    best_name = None
    best_k = None
    best_labels = None

    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = km.fit_predict(x_scaled)
        km_score = silhouette_score(
            x_scaled,
            km_labels,
            sample_size=min(len(x_scaled), 5000),
            random_state=42,
        )
        records.append({"Model": "KMeans", "k": k, "Silhouette": km_score})
        if km_score > best_score:
            best_score = km_score
            best_name = "KMeans"
            best_k = k
            best_labels = km_labels

        ag = AgglomerativeClustering(n_clusters=k)
        ag_labels = ag.fit_predict(x_scaled)
        ag_score = silhouette_score(
            x_scaled,
            ag_labels,
            sample_size=min(len(x_scaled), 5000),
            random_state=42,
        )
        records.append({"Model": "Agglomerative", "k": k, "Silhouette": ag_score})
        if ag_score > best_score:
            best_score = ag_score
            best_name = "Agglomerative"
            best_k = k
            best_labels = ag_labels

    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    out = X.copy()
    out["cluster"] = best_labels
    profile = out.groupby("cluster").mean(numeric_only=True).round(2)
    sil_df = pd.DataFrame(records)
    explained = float(pca.explained_variance_ratio_.sum())

    return sil_df, out, profile, x_pca, explained, best_name, best_k


def render_objectif1(df_jobs: pd.DataFrame):
    st.header("Competences et categories (aligne sur vos donnees jobs)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes", len(df_jobs))
    c2.metric(
        "Metiers uniques",
        df_jobs[JOB_TITLE_COL].nunique() if JOB_TITLE_COL in df_jobs.columns else "N/A",
    )
    c3.metric("Requetes uniques", df_jobs["Query"].nunique() if "Query" in df_jobs.columns else "N/A")

    if "Query" in df_jobs.columns:
        st.subheader("Top categories de requetes")
        plot_category_bar(df_jobs["Query"], "Nombre d'offres par categorie de requete")

    if "IT Skills" in df_jobs.columns:
        st.subheader("Top competences techniques (IT Skills)")
        it_top = skill_frequency(df_jobs, "IT Skills", top_n=15)
        st.dataframe(it_top, width='stretch', hide_index=True)
        if not it_top.empty:
            fig = px.bar(it_top.iloc[::-1], x="count", y="skill", orientation="h", title="IT Skills")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8", height=400)
            st.plotly_chart(fig, width='stretch')

    if "Soft Skills" in df_jobs.columns:
        st.subheader("Top soft skills")
        soft_top = skill_frequency(df_jobs, "Soft Skills", top_n=15)
        st.dataframe(soft_top, width='stretch', hide_index=True)
        if not soft_top.empty:
            fig = px.bar(soft_top.iloc[::-1], x="count", y="skill", orientation="h", title="Soft Skills")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8", height=400)
            st.plotly_chart(fig, width='stretch')


def render_objectif2(df_jobs: pd.DataFrame):
    st.header("Classification — comparaison de modeles (meme logique que le notebook)")
    try:
        bench, best_name, cm, labels = run_model_comparison(df_jobs)
    except ValueError as exc:
        st.error(f"Impossible de calculer la comparaison de modeles: {exc}")
        st.info(
            "Astuce: ce dataset contient beaucoup de classes rares. "
            "Le tableau de comparaison apparaitra des que le split sera possible."
        )
        return
    st.subheader("Comparaison des modeles")
    st.dataframe(bench.round(4), width='stretch', hide_index=True)
    fig = px.bar(
        bench,
        x="Modele",
        y=["Accuracy", "F1 Macro"],
        barmode="group",
        title="Accuracy et F1 macro par modele",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8", xaxis_tickangle=-30)
    st.plotly_chart(fig, width='stretch')

    st.subheader(f"Matrice de confusion — meilleur modele: {best_name}")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    st.dataframe(cm_df, width='stretch')


def render_objectif3(df_hr: pd.DataFrame):
    st.header("Clustering + PCA (comme objectif 3 dans le code)")
    sil_df, clustered_df, profile, X_pca, explained, best_name, best_k = run_clustering(df_hr)

    st.subheader("Comparaison des modeles de clustering (silhouette)")
    st.dataframe(sil_df.sort_values(["Model", "k"]).round(4), width='stretch', hide_index=True)
    pivot = sil_df.pivot(index="k", columns="Model", values="Silhouette")
    st.line_chart(pivot)

    st.success(f"Meilleur modele: {best_name} (k={best_k})")

    st.subheader("Projection PCA (2 composantes)")
    st.caption(f"Variance expliquee totale par PCA: {explained:.2%}")
    pca_plot = pd.DataFrame(
        {
            "PCA_1": X_pca[:, 0],
            "PCA_2": X_pca[:, 1],
            "cluster": clustered_df["cluster"].astype(str).values,
        }
    )
    st.scatter_chart(pca_plot, x="PCA_1", y="PCA_2", color="cluster")

    st.subheader("Profil moyen par cluster")
    st.dataframe(profile, width='stretch')


signals = scan_ml_code_signals(str(PROJECT_ROOT))
any_signal = any(
    signals[k]
    for k in ("skills_extraction", "classification", "clustering", "pca")
)

st.sidebar.title("Navigation")
st.sidebar.caption("Theme clair bleu et blanc. La page active est marquee par une ombre.")
page = st.sidebar.radio(
    "Choisir une vue:",
    (
        "Vue dynamique — selon le code du projet",
        "Objectif 1 — Extraction automatique des competences visees",
        "Objectif 2 — Classification automatique des formations",
        "Objectif 3 — Segmentation (Clustering + PCA)",
    ),
    label_visibility="visible",
)

st.title("Dashboard ML — Presentation alignee sur votre code")
st.caption(
    "Les blocs de la vue dynamique apparaissent si les memes idees "
    "(competences, classification, clustering, PCA) sont presentes dans vos notebooks/scripts, "
    "puis sexecutent sur les CSV du dossier comme dans lapplication."
)

df_jobs = load_csv(DATASET_JOBS)
df_hr = load_csv(DATASET_HR)

if page.startswith("Vue dynamique"):
    st.subheader("Detection dans vos fichiers (hors `app.py` et environnement virtuel)")
    det = pd.DataFrame(
        [
            {"Analyse": "Extraction / competences & texte", "Detectee": signals["skills_extraction"]},
            {"Analyse": "Classification / comparaison de modeles", "Detectee": signals["classification"]},
            {"Analyse": "Clustering (KMeans, Agglomeratif, ...)", "Detectee": signals["clustering"]},
            {"Analyse": "PCA / reduction de dimension", "Detectee": signals["pca"]},
        ]
    )
    st.dataframe(det, width='stretch', hide_index=True)

    if not any_signal:
        st.warning(
            "Aucun signal ML clair detecte dans les sources. "
            "Verifiez que vos notebooks `.ipynb` sont bien dans ce dossier. "
            "Affichage de secours : tout ce qui est possible avec les CSV presents."
        )

    show_skills = signals["skills_extraction"] or (not any_signal and df_jobs is not None)
    show_clf = signals["classification"] or (not any_signal and df_jobs is not None)
    show_cluster = (signals["clustering"] or signals["pca"]) or (not any_signal and df_hr is not None)

    if show_skills:
        if df_jobs is None:
            st.error(f"Fichier introuvable: `{DATASET_JOBS}` (necessaire pour cette section).")
        else:
            render_objectif1(df_jobs)

    if show_clf:
        if df_jobs is None:
            st.error(f"Fichier introuvable: `{DATASET_JOBS}` (necessaire pour la classification).")
        else:
            render_objectif2(df_jobs)

    if show_cluster:
        if df_hr is None:
            st.error(f"Fichier introuvable: `{DATASET_HR}` (necessaire pour clustering / PCA).")
        else:
            render_objectif3(df_hr)

elif page.startswith("Objectif 1"):
    if df_jobs is None:
        st.error(f"Fichier introuvable: {DATASET_JOBS}")
    else:
        render_objectif1(df_jobs)

elif page.startswith("Objectif 2"):
    if df_jobs is None:
        st.error(f"Fichier introuvable: {DATASET_JOBS}")
    else:
        render_objectif2(df_jobs)

elif page.startswith("Objectif 3"):
    if df_hr is None:
        st.error(f"Fichier introuvable: {DATASET_HR}")
    else:
        render_objectif3(df_hr)
