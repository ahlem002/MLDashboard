import ast
import json
import os
import re
from pathlib import Path
from objectif3_page import render_objectif3_page
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample


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

DATASET_JOBS = os.environ.get("ML_OBJECTIF1_DATASET", "activities.xlsx").strip() or "activities.xlsx"
DATASET_HR = "Cleaned_HR_Data_Analysis.csv"
JOB_TITLE_COL = os.environ.get("ML_OBJECTIF1_TITLE_COL", "job_title").strip() or "job_title"

# --- Objectif 2 (classification): edit here when you change notebook / CSV ---
# Relative paths are resolved from the project folder (same directory as app.py).
# Optional overrides via environment: ML_OBJECTIF2_CSV, ML_OBJECTIF2_TEXT_COL, ML_OBJECTIF2_LABEL_COL
DATASET_CLASSIFICATION = os.environ.get(
    "ML_OBJECTIF2_CSV",
    "JobsDatasetProcessed (2).csv",
).strip() or "JobsDatasetProcessed (2).csv"
# crisp_v2 = meme logique que CRISP_DM_Classification_V2.ipynb (clean_text, category, resample, SVM)
# legacy = ancienne comparaison multi-modeles sur deux colonnes texte / label au choix
OBJECTIF2_PIPELINE = (
    os.environ.get("ML_OBJECTIF2_PIPELINE", "crisp_v2").strip().lower() or "crisp_v2"
)
CLASSIFICATION_TEXT_COL = (os.environ.get("ML_OBJECTIF2_TEXT_COL") or "").strip() or None
CLASSIFICATION_LABEL_COL = (os.environ.get("ML_OBJECTIF2_LABEL_COL") or "").strip() or None

TEXT_COLUMN_CANDIDATES = (
    "clean_text",
    "combined_text",
    "Description",
    "text",
    "requirements_and_role",
    "job_requirements",
    "body",
    "content",
    "source_text",
)
LABEL_COLUMN_CANDIDATES = (
    "category",
    JOB_TITLE_COL,
    "label",
    "target",
    "y",
    "formation",
    "class",
    "Job_Category",
)

PROJECT_ROOT = Path(__file__).resolve().parent
SKIP_SCAN_NAMES = {"app.py"}
SKIP_DIR_NAMES = {".venv", "venv", "__pycache__", ".git", "node_modules"}


# ─── spaCy NLP for skill extraction ───
@st.cache_resource
def load_spacy_nlp_with_skills():
    """Load spaCy model and configure EntityRuler with skill patterns (matching notebook)."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except (IOError, OSError):
        # Fallback if model is not installed
        st.warning("spaCy model not found. Using keyword-based skill extraction as fallback.")
        return None
    
    # Remove existing entity_ruler if present
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    
    # Create skill patterns (matching Objectif1_Extraction_Competences.ipynb)
    SKILL_PATTERNS = [
        # ── TECHNICAL SKILLS ──
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'python'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'sql'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'java'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'javascript'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'r'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'c++'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'c#'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'react'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'angular'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'vue'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'node'}, {'LOWER': 'js'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'aws'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'azure'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'docker'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'kubernetes'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'git'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'linux'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'machine'}, {'LOWER': 'learning'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'deep'}, {'LOWER': 'learning'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'data'}, {'LOWER': 'analysis'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'data'}, {'LOWER': 'science'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'data'}, {'LOWER': 'engineering'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'artificial'}, {'LOWER': 'intelligence'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'natural'}, {'LOWER': 'language'}, {'LOWER': 'processing'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'nlp'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'tableau'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'power'}, {'LOWER': 'bi'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'tensorflow'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'pytorch'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'scikit'}, {'LOWER': 'learn'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'excel'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'photoshop'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'autocad'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'networking'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'cybersecurity'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'cloud'}, {'LOWER': 'computing'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'devops'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'api'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'restful'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'microservices'}]},
        {'label': 'TECHNICAL_SKILL', 'pattern': [{'LOWER': 'sap'}]},
        
        # ── MANAGERIAL SKILLS ──
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'leadership'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'management'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'planning'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'strategic'}, {'LOWER': 'planning'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'project'}, {'LOWER': 'management'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'team'}, {'LOWER': 'management'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'team'}, {'LOWER': 'leadership'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'agile'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'scrum'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'kanban'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'budgeting'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'budget'}, {'LOWER': 'management'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'vendor'}, {'LOWER': 'management'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'stakeholder'}, {'LOWER': 'management'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'change'}, {'LOWER': 'management'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'risk'}, {'LOWER': 'management'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'performance'}, {'LOWER': 'management'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'decision'}, {'LOWER': 'making'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'coaching'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'mentoring'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'negotiation'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'coordination'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'recruitment'}]},
        {'label': 'MANAGERIAL_SKILL', 'pattern': [{'LOWER': 'training'}]},
        
        # ── SOFT SKILLS ──
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'communication'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'teamwork'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'problem'}, {'LOWER': 'solving'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'critical'}, {'LOWER': 'thinking'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'adaptability'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'time'}, {'LOWER': 'management'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'creativity'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'collaboration'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'interpersonal'}, {'LOWER': 'skills'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'presentation'}, {'LOWER': 'skills'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'analytical'}, {'LOWER': 'skills'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'attention'}, {'LOWER': 'to'}, {'LOWER': 'detail'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'multitasking'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'flexibility'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'initiative'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'customer'}, {'LOWER': 'service'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'relationship'}, {'LOWER': 'management'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'emotional'}, {'LOWER': 'intelligence'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'work'}, {'LOWER': 'ethic'}]},
        {'label': 'SOFT_SKILL', 'pattern': [{'LOWER': 'public'}, {'LOWER': 'speaking'}]},
    ]
    
    # Add entity ruler with patterns
    ruler = nlp.add_pipe('entity_ruler', before='ner')
    ruler.add_patterns(SKILL_PATTERNS)
    return nlp


def extract_skills_from_text(text, nlp):
    """Extract skills using spaCy EntityRuler or keyword matching (matching notebook methodology)."""
    if not isinstance(text, str) or len(text.strip()) < 5:
        return {"TECHNICAL_SKILL": [], "MANAGERIAL_SKILL": [], "SOFT_SKILL": []}
    
    if nlp is not None:
        # Use spaCy if available
        doc = nlp(text[:5000])  # Limit for performance
        extracted = {"TECHNICAL_SKILL": set(), "MANAGERIAL_SKILL": set(), "SOFT_SKILL": set()}
        
        for ent in doc.ents:
            if ent.label_ in extracted:
                extracted[ent.label_].add(ent.text.lower().strip())
        
        return {k: sorted(v) for k, v in extracted.items()}
    else:
        # Fallback to keyword matching if spacy not available
        text_lower = text.lower()
        technical_keywords = {
            "python", "sql", "java", "c++", "c#", "javascript", "aws", "azure", "docker",
            "kubernetes", "ml", "ai", "data", "api", "linux", "cloud", "etl", "tableau",
            "tensorflow", "pytorch", "scikit", "react", "angular", "vue", "git", "devops",
            "excel", "photoshop", "autocad", "networking", "cybersecurity", "microservices", "sap"
        }
        managerial_keywords = {
            "manage", "manager", "lead", "leadership", "project", "strategy", "planning", "budget",
            "stakeholder", "coordination", "scrum", "agile", "roadmap", "decision", "coaching",
            "mentoring", "negotiation", "recruitment", "training"
        }
        soft_keywords = {
            "communication", "teamwork", "collaboration", "problem solving", "adaptability", "creativity",
            "critical thinking", "time management", "presentation", "negotiation", "empathy",
            "flexibility", "initiative", "customer service", "emotional intelligence", "work ethic", "public speaking"
        }
        
        extracted = {"TECHNICAL_SKILL": set(), "MANAGERIAL_SKILL": set(), "SOFT_SKILL": set()}
        
        for word in text_lower.split():
            word_clean = word.lower().strip(".,!?;:")
            if word_clean in technical_keywords:
                extracted["TECHNICAL_SKILL"].add(word_clean)
            elif word_clean in managerial_keywords:
                extracted["MANAGERIAL_SKILL"].add(word_clean)
            elif word_clean in soft_keywords:
                extracted["SOFT_SKILL"].add(word_clean)
        
        return {k: sorted(v) for k, v in extracted.items()}


def resolve_dataset_path(path: str) -> str:
    """Turn a project-relative or absolute path into a concrete path string."""
    p = (path or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return str(PROJECT_ROOT / p)


@st.cache_data
def load_csv(path: str):
    resolved = resolve_dataset_path(path)
    if resolved and os.path.exists(resolved):
        ext = Path(resolved).suffix.lower()
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(resolved)
        return pd.read_csv(resolved, engine="python", on_bad_lines="skip")
    return None


def pick_classification_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Choisit colonnes texte / label (mode legacy) selon la config ou l'ordre de priorite."""
    text_col = CLASSIFICATION_TEXT_COL
    label_col = CLASSIFICATION_LABEL_COL
    if text_col and text_col not in df.columns:
        raise ValueError(f"Colonne texte inconnue: {text_col!r}. Colonnes: {list(df.columns)}")
    if label_col and label_col not in df.columns:
        raise ValueError(f"Colonne label inconnue: {label_col!r}. Colonnes: {list(df.columns)}")
    if not text_col:
        for c in TEXT_COLUMN_CANDIDATES:
            if c in df.columns:
                text_col = c
                break
    if not label_col:
        for c in LABEL_COLUMN_CANDIDATES:
            if c in df.columns:
                label_col = c
                break
    if not text_col or not label_col:
        raise ValueError(
            "Impossible de deviner les colonnes texte/label. "
            "Definissez ML_OBJECTIF2_TEXT_COL et ML_OBJECTIF2_LABEL_COL "
            f"(colonnes disponibles: {list(df.columns)})"
        )
    return text_col, label_col


def _crisp_clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _crisp_categoriser(text: str) -> str:
    tech = ["python", "java", "sql", "data", "developer", "engineering", "machine learning"]
    mg = ["manager", "lead", "project", "scrum", "director", "strategy"]
    if any(w in text for w in tech):
        return "technique"
    if any(w in text for w in mg):
        return "managerial"
    return "soft_skills"


def prepare_crisp_v2_balanced_df(df: pd.DataFrame) -> pd.DataFrame:
    """Meme preparation que CRISP_DM_Classification_V2.ipynb (sans dependre du CSV pre-calcule)."""
    df = df.copy()
    cols = df.columns
    text_cols = [c for c in cols if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    if not text_cols:
        raise ValueError("Aucune colonne texte (object/string) pour construire combined_text.")
    df["combined_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)
    df["clean_text"] = df["combined_text"].apply(_crisp_clean_text)
    df["category"] = df["clean_text"].apply(_crisp_categoriser)
    vc = df["category"].value_counts()
    if len(vc) < 2:
        raise ValueError(
            f"Il faut au moins 2 classes apres categorisation. Repartition: {vc.to_dict()}"
        )
    df_max = int(vc.max())
    dfs = []
    for cat in df["category"].unique():
        df_cat = df[df["category"] == cat]
        df_cat_up = resample(df_cat, replace=True, n_samples=df_max, random_state=42)
        dfs.append(df_cat_up)
    return pd.concat(dfs).sample(frac=1, random_state=42)


@st.cache_data
def run_crisp_v2_model_comparison(resolved_csv_path: str):
    """Aligne sur CRISP_DM_Classification_V2.ipynb: lecture robuste, TF-IDF 3000, SVM."""
    df = pd.read_csv(resolved_csv_path, engine="python", on_bad_lines="skip")
    df_balanced = prepare_crisp_v2_balanced_df(df)

    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vectorizer.fit_transform(df_balanced["clean_text"])
    encoder = LabelEncoder()
    y = encoder.fit_transform(df_balanced["category"])
    class_names = encoder.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {"SVM": SVC(class_weight="balanced")}
    rows = []
    best_name = None
    best_pred = None
    best_acc = -1.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1m = f1_score(y_test, pred, average="macro", zero_division=0)
        rows.append({"Modele": name, "Accuracy": acc, "F1 Macro": f1m})
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pred = pred

    bench = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    labels_idx = sorted(set(y_test) | set(best_pred))
    cm = confusion_matrix(y_test, best_pred, labels=labels_idx)
    labels_human = [class_names[i] for i in labels_idx]
    return bench, best_name, cm, labels_human


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
def run_legacy_model_comparison(df_clf: pd.DataFrame, text_col: str, label_col: str):
    """Comparaison multi-modeles sur deux colonnes (mode legacy, ex. Description + Job Title)."""
    data = df_clf[[text_col, label_col]].dropna().copy()
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


@st.cache_data(show_spinner=False)
def prepare_objectif1_analysis(df_jobs: pd.DataFrame, text_col: str | None):
    empty_skills = pd.DataFrame(columns=["row_idx", "skill", "type"])
    result = {
        "raw_text": None,
        "clean_text": None,
        "skills_df": empty_skills,
        "sample_raw": "",
        "sample_clean": "",
        "sample_skills": empty_skills,
        "error": None,
    }

    if text_col is None or text_col not in df_jobs.columns:
        return result

    try:
        raw_text = df_jobs[text_col].fillna("").astype(str)
        clean_text = raw_text.apply(_crisp_clean_text)

        nlp = load_spacy_nlp_with_skills()
        skill_records = []
        for idx, text in enumerate(clean_text):
            try:
                skills = extract_skills_from_text(text, nlp)
            except Exception:
                continue
            for skill_type, skill_list in skills.items():
                for skill in skill_list:
                    skill_records.append({"row_idx": idx, "skill": skill, "type": skill_type})

        skills_df = pd.DataFrame(skill_records) if skill_records else empty_skills.copy()
        sample_idx = int(raw_text.str.len().idxmax()) if (raw_text.str.len() > 0).any() else int(df_jobs.index[0])
        sample_skills = skills_df[skills_df["row_idx"] == sample_idx] if not skills_df.empty else empty_skills.copy()

        result.update(
            {
                "raw_text": raw_text,
                "clean_text": clean_text,
                "skills_df": skills_df,
                "sample_raw": raw_text.loc[sample_idx],
                "sample_clean": clean_text.loc[sample_idx],
                "sample_skills": sample_skills,
            }
        )
    except Exception as exc:
        result["error"] = str(exc)

    return result


def render_objectif1(df_jobs: pd.DataFrame):
    st.header("Objectif 1 — Dataset Overview, Text Processing, Skills Analytics")

    def _first_col(candidates: list[str]) -> str | None:
        for col in candidates:
            if col in df_jobs.columns:
                return col
        return None

    text_col = _first_col([
        "requirements_and_role",
        "job_requirements",
        "Description",
        "description",
        "combined_text",
        "text",
    ])
    # If no specific text column found, try any object/string column
    if text_col is None:
        for col in df_jobs.columns:
            if df_jobs[col].dtype == "object" or pd.api.types.is_string_dtype(df_jobs[col]):
                text_col = col
                break
    
    skills_col = _first_col(["skills_required", "IT Skills", "Soft Skills", "skills", "Skills"])
    category_col = _first_col(["job_category", "Query", "category", "Job_Category"])

    total_rows = len(df_jobs)
    total_cols = df_jobs.shape[1]
    missing_values = int(df_jobs.isna().sum().sum())
    n_categories = int(df_jobs[category_col].nunique()) if category_col else 0
    analysis = prepare_objectif1_analysis(df_jobs, text_col)
    raw_text = analysis["raw_text"]
    clean_text = analysis["clean_text"]
    skills_df = analysis["skills_df"]
    sample_raw = analysis["sample_raw"]
    sample_clean = analysis["sample_clean"]
    sample_skills = analysis["sample_skills"]

    def _metric_card(title: str, value: str, accent: str) -> str:
        return f"""
        <div style="background:#ffffff;border:1px solid #dbe8ff;border-left:6px solid {accent};border-radius:14px;padding:18px 20px;box-shadow:0 10px 22px rgba(33,88,185,0.08);">
          <div style="font-size:14px;color:#486581;margin-bottom:8px;">{title}</div>
          <div style="font-size:34px;font-weight:800;color:#102a43;line-height:1.1;">{value}</div>
        </div>
        """

    st.subheader("1. Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_metric_card("Total rows", f"{total_rows:,}", "#2563eb"), unsafe_allow_html=True)
    with c2:
        st.markdown(_metric_card("Columns detected", f"{total_cols:,}", "#0ea5e9"), unsafe_allow_html=True)
    with c3:
        st.markdown(_metric_card("Missing values", f"{missing_values:,}", "#ef4444"), unsafe_allow_html=True)
    with c4:
        st.markdown(_metric_card("Job categories", f"{n_categories:,}", "#7c3aed"), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("2. Text Processing + Skill Extraction")

    if text_col is None:
        st.warning("⚠️ No raw job-description text column detected. Available columns: " + str(list(df_jobs.columns)))
        st.info("Please ensure your dataset has a text/description column.")
        return

    if analysis["error"]:
        st.warning(f"Text processing used cached fallback with an issue: {analysis['error']}")

    if not isinstance(raw_text, pd.Series) or not isinstance(clean_text, pd.Series):
        st.warning("No text available for deeper analysis.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**Raw job description**")
        st.markdown(
            f"""
            <div style="background:#ffffff;border:1px solid #dbe8ff;border-radius:14px;padding:16px;min-height:260px;color:#102a43;line-height:1.7;font-size:15px;white-space:pre-wrap;">{sample_raw[:1800]}</div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**Cleaned text + highlighted skills**")
        st.markdown(
            f"""
            <div style="background:#ffffff;border:1px solid #dbe8ff;border-radius:14px;padding:16px;min-height:180px;color:#102a43;line-height:1.7;font-size:15px;white-space:pre-wrap;">{sample_clean[:1200]}</div>
            """,
            unsafe_allow_html=True,
        )
        if not sample_skills.empty:
            color_map = {"TECHNICAL_SKILL": "#2563eb", "MANAGERIAL_SKILL": "#7c3aed", "SOFT_SKILL": "#0d9488"}
            chips = []
            for _, row in sample_skills.head(20).iterrows():
                color = color_map.get(row["type"], "#334155")
                # Display friendly names
                type_display = row["type"].replace("_SKILL", "").replace("_", " ").title()
                chips.append(
                    f"<span style='display:inline-block;margin:4px 6px 0 0;padding:6px 10px;border-radius:999px;background:{color}1A;border:1px solid {color}66;color:{color};font-size:13px;font-weight:700'>{row['skill']} ({type_display})</span>"
                )
            st.markdown("".join(chips), unsafe_allow_html=True)
        else:
            st.info("No extracted skills found for this sample row.")

    st.markdown("---")
    st.subheader("3. Skills Analytics")

    if skills_df.empty:
        st.warning("⚠️ No skills extracted yet. This may indicate issues with text processing.")
        st.info("Skipping analytics sections...")

    if not skills_df.empty:
        block_a, block_b, block_c = st.columns(3)

        with block_a:
            st.markdown("**A. Distribution — #skills per job**")
            counts = skills_df.groupby("row_idx").size().rename("skill_count")
            all_counts = pd.Series(0, index=df_jobs.index, dtype=int)
            all_counts.loc[counts.index] = counts.values
            hist_fig = px.histogram(
                all_counts,
                nbins=20,
                title="#skills per job",
                labels={"value": "Skills per job", "count": "Jobs"},
            )
            hist_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8", height=360)
            st.plotly_chart(hist_fig, width="stretch")

        with block_b:
            st.markdown("**B. Top Skills — Top 15 skills (per category)**")
            skill_type_choice = st.selectbox(
                "Skill category",
                ["Technical", "Managerial", "Soft"],
                key="obj1_skill_type_choice",
            )
            # Map friendly names to actual labels
            label_map = {"Technical": "TECHNICAL_SKILL", "Managerial": "MANAGERIAL_SKILL", "Soft": "SOFT_SKILL"}
            actual_label = label_map[skill_type_choice]
            
            top_skills = (
                skills_df[skills_df["type"] == actual_label]
                .groupby("skill")
                .size()
                .sort_values(ascending=False)
                .head(15)
                .reset_index(name="count")
            )
            if top_skills.empty:
                st.info("No skills in this category.")
            else:
                top_fig = px.bar(
                    top_skills.iloc[::-1],
                    x="count",
                    y="skill",
                    orientation="h",
                    title=f"Top 15 {skill_type_choice} skills",
                    color_discrete_sequence=["#2563eb"],
                )
                top_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8", height=360)
                st.plotly_chart(top_fig, width="stretch")

        with block_c:
            st.markdown("**C. Global Split**")
            split_df = (
                skills_df["type"]
                .value_counts()
                .reindex(["TECHNICAL_SKILL", "MANAGERIAL_SKILL", "SOFT_SKILL"], fill_value=0)
                .reset_index()
            )
            split_df.columns = ["Type", "Count"]
            # Map back to friendly names for display
            split_df["Type_Display"] = split_df["Type"].map({
                "TECHNICAL_SKILL": "Technical",
                "MANAGERIAL_SKILL": "Managerial",
                "SOFT_SKILL": "Soft"
            })
            pie_fig = px.pie(
                split_df,
                names="Type_Display",
                values="Count",
                title="Technical vs Managerial vs Soft",
                color="Type_Display",
                color_discrete_map={"Technical": "#2563eb", "Managerial": "#7c3aed", "Soft": "#0d9488"},
            )
            pie_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8", height=360)
            st.plotly_chart(pie_fig, width="stretch")

    st.markdown("---")
    st.subheader("4. Model Training + Benchmark")

    if text_col is None or clean_text is None or len(clean_text) == 0:
        st.warning("⚠️ Not enough text data for model training.")
    else:
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.ensemble import RandomForestClassifier as RFC

        # Build classification dataset from extracted skills (matching notebook methodology)
        if not skills_df.empty and len(clean_text) > 0:
            clf_rows = []
            
            # Map each skill with its source text context
            for _, row in skills_df.iterrows():
                row_idx = int(row["row_idx"])
                if row_idx < len(clean_text):
                    source_text = clean_text.iloc[row_idx][:500] if pd.notna(clean_text.iloc[row_idx]) else ""
                    if source_text.strip():
                        clf_rows.append({
                            "skill": row["skill"].lower().strip() if pd.notna(row["skill"]) else "",
                            "label": row["type"] if pd.notna(row["type"]) else "Technical",
                            "source_text": source_text
                        })
            
            if clf_rows:
                clf_df = pd.DataFrame(clf_rows)
                clf_df = clf_df[clf_df["skill"].str.len() > 0]  # Remove empty skills
                clf_df = clf_df.drop_duplicates(subset=["skill", "label"]).reset_index(drop=True)
                
                if len(clf_df) >= 20:  # Need minimum samples for training
                    st.markdown("**Model Comparison (Skill Classification)**")
                    
                    # TF-IDF vectorization on source context
                    vec = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
                    X = vec.fit_transform(clf_df["source_text"])
                    y = clf_df["label"]
                    
                    # Train/test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    
                    # Define models (matching notebook)
                    models_bench = {
                        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3, metric="cosine"),
                        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
                        "Logistic Regression": LR(max_iter=500, random_state=42),
                        "Naive Bayes": MultinomialNB(alpha=0.1),
                        "Random Forest": RFC(n_estimators=100, random_state=42, n_jobs=-1),
                    }
                    
                    bench_rows = []
                    best_score = -1
                    best_model_name = None
                    best_y_pred = None
                    
                    for model_name, model in models_bench.items():
                        try:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
                            bench_rows.append({"Model": model_name, "Accuracy": acc, "F1 Macro": f1m})
                            if f1m > best_score:
                                best_score = f1m
                                best_model_name = model_name
                                best_y_pred = y_pred
                        except Exception:
                            pass
                    
                    if bench_rows:
                        bench_df = pd.DataFrame(bench_rows).sort_values("F1 Macro", ascending=False).reset_index(drop=True)
                        bench_col1, bench_col2 = st.columns(2)
                        
                        with bench_col1:
                            fig_bench = px.bar(
                                bench_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                                x="Model",
                                y="Score",
                                color="Metric",
                                barmode="group",
                                title="Model Comparison (Accuracy & F1 Macro)",
                                color_discrete_map={"Accuracy": "#2563eb", "F1 Macro": "#059669"},
                            )
                            fig_bench.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8")
                            st.plotly_chart(fig_bench, width="stretch")
                        
                        with bench_col2:
                            st.markdown(f"**Best Model: {best_model_name}** (F1 Macro = {best_score:.3f})")
                            st.dataframe(bench_df.round(3), width="stretch", hide_index=True)
                        
                        if best_y_pred is not None:
                            st.markdown("**Confusion Matrix (Best Model)**")
                            # Get unique labels from training data
                            unique_labels = sorted(y_test.unique())
                            cm = confusion_matrix(y_test, best_y_pred, labels=unique_labels)
                            cm_df = pd.DataFrame(
                                cm,
                                index=unique_labels,
                                columns=unique_labels
                            )
                            cm_fig = px.imshow(
                                cm_df,
                                text_auto=True,
                                aspect="auto",
                                title=f"Confusion Matrix - {best_model_name}",
                                color_continuous_scale="Blues",
                            )
                            cm_fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="#f0f4f8",
                                xaxis_title="Predicted",
                                yaxis_title="Actual",
                            )
                            st.plotly_chart(cm_fig, width="stretch")
                else:
                    st.info(f"Not enough unique skills for training ({len(clf_df)} skills, need at least 20).")
            else:
                st.info("No valid skills extracted for model training.")
        else:
            st.info("No skill data available for model training.")

    st.markdown("---")
    st.subheader("5. Output / Results Table")

    if not skills_df.empty:
        output_rows = []
        for idx, row in df_jobs.iterrows():
            title = str(row.get(text_col, "N/A")) if text_col and text_col in row else "N/A"
            title_short = str(title)[:60] if title != "N/A" else "N/A"
            skills_in_row = skills_df[skills_df["row_idx"] == idx]
            # Fix: Use correct skill type labels (TECHNICAL_SKILL, not "Technical")
            tech_skills = ", ".join(skills_in_row[skills_in_row["type"] == "TECHNICAL_SKILL"]["skill"].unique())
            mgr_skills = ", ".join(skills_in_row[skills_in_row["type"] == "MANAGERIAL_SKILL"]["skill"].unique())
            soft_skills = ", ".join(skills_in_row[skills_in_row["type"] == "SOFT_SKILL"]["skill"].unique())
            total_skills = len(skills_in_row)
            output_rows.append({
                "Job Title": title_short,
                "Technical Skills": tech_skills if tech_skills else "—",
                "Managerial Skills": mgr_skills if mgr_skills else "—",
                "Soft Skills": soft_skills if soft_skills else "—",
                "Total Skills": total_skills,
            })

        if output_rows:
            output_df = pd.DataFrame(output_rows).head(1000)
            st.dataframe(output_df, width="stretch", hide_index=True)

            csv_buffer = output_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Export CSV",
                data=csv_buffer,
                file_name="objectif1_skills_analysis.csv",
                mime="text/csv",
            )
        else:
            st.info("No skills data available to display.")
    else:
        st.info("No skills extracted. Check the Text Processing section above for errors.")


def render_objectif2():
    st.header("Objectif 2 — Text Processing et preparation des donnees")

    dataset_candidates = [DATASET_CLASSIFICATION, DATASET_JOBS]
    dataset_path = None
    dataset_name = None
    df = None
    for candidate in dataset_candidates:
        loaded = load_csv(candidate)
        resolved = resolve_dataset_path(candidate)
        if loaded is not None:
            df = loaded
            dataset_path = resolved
            dataset_name = candidate
            break

    if df is None:
        st.error("Impossible de charger le dataset de l'objectif 2.")
        return

    text_columns = [
        col
        for col in df.columns
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col])
    ]
    if not text_columns:
        st.error("Aucune colonne texte detectee pour construire l'analyse.")
        return

    display_text_columns = []
    for col in text_columns:
        series = df[col].fillna("").astype(str)
        avg_len = float(series.str.len().replace(0, np.nan).mean() or 0.0)
        sample = next((val.strip() for val in series.tolist() if val.strip()), "")
        display_text_columns.append(
            {
                "Colonne": col,
                "Moy. longueur": round(avg_len, 1),
                "Exemple": sample[:90],
            }
        )

    raw_text = (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    working_df = df.loc[raw_text.str.len() > 0].copy()
    working_df["raw_text"] = raw_text.loc[working_df.index]
    working_df["clean_text"] = working_df["raw_text"].apply(_crisp_clean_text)
    working_df["category"] = working_df["clean_text"].apply(_crisp_categoriser)

    st.caption(f"Dataset utilise: `{dataset_name}`")

    st.subheader("Data Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", f"{len(working_df):,}")
    m2.metric("Columns", f"{df.shape[1]:,}")
    m3.metric("Text columns detectees", f"{len(text_columns):,}")

    st.success(f"Your dataset contains {len(text_columns)} text columns and {len(working_df)} entries.")

    st.markdown("**Dataset preview (first rows)**")
    st.dataframe(df.head(10), width="stretch", hide_index=True)

    st.markdown("**Detected text columns (auto-detection feature)**")
    st.dataframe(pd.DataFrame(display_text_columns), width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Number of rows / columns**")
        st.write(f"{len(working_df):,} rows and {df.shape[1]:,} columns")
    with right:
        st.markdown("**Dataset source**")
        st.write(dataset_path)

    st.markdown("#### Missing values")
    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing values"]
    missing = missing[missing["Missing values"] > 0]
    if missing.empty:
        st.info("No missing values detected.")
    else:
        missing_fig = px.bar(
            missing.sort_values("Missing values", ascending=True),
            x="Missing values",
            y="Column",
            orientation="h",
            title="Bar chart of missing values",
        )
        missing_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8")
        st.plotly_chart(missing_fig, width="stretch")

    st.markdown("#### Distribution of raw text length")
    raw_lengths = working_df["raw_text"].str.len()
    length_fig = px.histogram(
        raw_lengths,
        nbins=30,
        title="Distribution of raw text length",
        labels={"value": "Raw text length"},
    )
    length_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8")
    st.plotly_chart(length_fig, width="stretch")

    st.markdown("---")
    st.subheader("Text Processing")
    st.caption("Show transformation clearly from raw text to cleaned text.")

    example_row = working_df.iloc[0]
    sample_raw = example_row["raw_text"]
    sample_clean = example_row["clean_text"]

    def _text_card(title: str, text: str, accent: str) -> str:
        return f"""
        <div style="background:#ffffff;border:1px solid #dbe8ff;border-left:6px solid {accent};border-radius:16px;padding:24px;box-shadow:0 10px 26px rgba(33,88,185,0.08);height:100%;">
          <div style="font-size:22px;font-weight:800;color:{accent};margin-bottom:12px;">{title}</div>
          <div style="white-space:pre-wrap;line-height:1.75;color:#102a43;font-size:16px;">{text[:1200]}</div>
        </div>
        """

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_text_card("Before cleaning", sample_raw, "#2563eb"), unsafe_allow_html=True)
    with c2:
        st.markdown(_text_card("After cleaning", sample_clean, "#059669"), unsafe_allow_html=True)

    # Calculate raw tokens (all tokens including stop words)
    raw_tokens_per_doc = working_df["raw_text"].str.lower().str.findall(r"\b[a-zA-Z]{2,}\b").apply(len)
    
    # Calculate clean tokens (excluding stop words to show real cleaning impact)
    def count_tokens_without_stopwords(text):
        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return len([t for t in tokens if t not in ENGLISH_STOP_WORDS])
    
    clean_tokens_per_doc = working_df["clean_text"].apply(count_tokens_without_stopwords)
    
    token_metrics = st.columns(4)
    token_metrics[0].metric("Raw tokens / doc", f"{raw_tokens_per_doc.mean():.1f}")
    token_metrics[1].metric("Clean tokens / doc (excl. stopwords)", f"{clean_tokens_per_doc.mean():.1f}")
    word_pattern = r"\b[a-zA-Z]{2,}\b"

    raw_text_all = " ".join(working_df["raw_text"].str.lower().tolist())
    unique_raw_words = len(set(re.findall(word_pattern, raw_text_all)))

    token_metrics[2].metric(
     "Unique raw words",
      f"{unique_raw_words:,}",
    )
    
    # Unique clean words: excluding stop words to show real cleaning impact
    clean_text_all = " ".join(working_df["clean_text"].str.lower().tolist())
    all_clean_words = re.findall(word_pattern, clean_text_all)
    unique_clean_words = len(set([w for w in all_clean_words if w not in ENGLISH_STOP_WORDS]))
    
    token_metrics[3].metric(
    "Unique clean words (excl. stopwords)",
    f"{unique_clean_words:,}",
    )
   

    cleaned_tokens = [
        token
        for text in working_df["clean_text"].str.lower().tolist()
        for token in re.findall(r"\b[a-zA-Z]{2,}\b", text)
        if token not in ENGLISH_STOP_WORDS
    ]
    if cleaned_tokens:
        from collections import Counter

        top_words = pd.DataFrame(Counter(cleaned_tokens).most_common(15), columns=["Word", "Count"])
        st.markdown("**Most frequent words**")
        word_fig = px.bar(
            top_words.sort_values("Count", ascending=True),
            x="Count",
            y="Word",
            orientation="h",
            title="Most frequent words",
        )
        word_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8")
        st.plotly_chart(word_fig, width="stretch")
    else:
        st.info("Not enough tokens to build the word frequency chart.")

    st.markdown("**Top keywords extracted using TF-IDF**")
    tfidf_source = working_df["clean_text"].fillna("").astype(str)
    tfidf_source = tfidf_source[tfidf_source.str.strip() != ""]
    if len(tfidf_source) >= 2:
        vectorizer = TfidfVectorizer(max_features=20, stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(tfidf_source)
        scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())
        top_idx = np.argsort(scores)[::-1][:12]
        tfidf_df = pd.DataFrame({"Keyword": terms[top_idx], "Score": scores[top_idx]})
        tfidf_fig = px.bar(
            tfidf_df.sort_values("Score", ascending=True),
            x="Score",
            y="Keyword",
            orientation="h",
            title="Top keywords extracted using TF-IDF",
        )
        tfidf_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8")
        st.plotly_chart(tfidf_fig, width="stretch")
    else:
        st.info("Not enough cleaned text to compute TF-IDF keywords.")

    st.markdown("---")
    st.subheader("Auto Labeling")
    st.caption("Categories generated by the categoriser function.")

    cat_order = ["technique", "managerial", "soft_skills"]
    cat_counts = (
        working_df["category"]
        .value_counts()
        .reindex(cat_order, fill_value=0)
        .reset_index()
    )
    cat_counts.columns = ["category", "count"]

    pie_fig = px.pie(
        cat_counts,
        names="category",
        values="count",
        title="Pie chart of categories",
        color="category",
        color_discrete_map={
            "technique": "#2563eb",
            "managerial": "#0ea5e9",
            "soft_skills": "#14b8a6",
        },
    )
    pie_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f8")
    st.plotly_chart(pie_fig, width="stretch")

    st.markdown("---")
    st.subheader("Model Training")

    st.write("Selected model: Support Vector Machine (SVM)")
    st.write("Training/test split: 80/20")
    st.write("Class balancing: Upsampling minority classes (matching CRISP-DM notebook)")

    train_df = working_df[["clean_text", "category"]].copy()
    train_df = train_df[train_df["clean_text"].str.strip() != ""]

    if len(train_df) < 10 or train_df["category"].nunique() < 2:
        st.info("Not enough labeled samples to train SVM. Need at least 10 rows and 2 categories.")
    else:
        # Apply class balancing (upsampling) like CRISP-DM notebook
        df_max = int(train_df["category"].value_counts().max())
        dfs_balanced = []
        for cat in train_df["category"].unique():
            df_cat = train_df[train_df["category"] == cat]
            df_cat_up = resample(df_cat, replace=True, n_samples=df_max, random_state=42)
            dfs_balanced.append(df_cat_up)
        train_df_balanced = pd.concat(dfs_balanced).sample(frac=1, random_state=42).reset_index(drop=True)
        
        x = train_df_balanced["clean_text"].astype(str)
        y = train_df_balanced["category"].astype(str)

        test_size = max(1, int(np.ceil(0.2 * len(train_df_balanced))))
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y,
            )
        except ValueError:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=test_size,
                random_state=42,
                stratify=None,
            )

        svm_vec = TfidfVectorizer(max_features=3000, stop_words="english")
        x_train_vec = svm_vec.fit_transform(x_train)
        x_test_vec = svm_vec.transform(x_test)

        svm_model = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")
        svm_model.fit(x_train_vec, y_train)
        y_pred = svm_model.predict(x_test_vec)

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Train rows (balanced)", f"{len(x_train):,}")
        k2.metric("Test rows", f"{len(x_test):,}")
        k3.metric("Accuracy", f"{acc:.3f}")
        k4.metric("F1 Macro", f"{f1m:.3f}")

        st.markdown("---")
        st.subheader("Evaluation")

        st.markdown(
            f"""
            <div style="background:#ffffff;border:1px solid #dbe8ff;border-left:8px solid #2563eb;border-radius:16px;padding:22px 26px;box-shadow:0 10px 24px rgba(33,88,185,0.08);max-width:380px;">
              <div style="font-size:14px;color:#486581;margin-bottom:8px;">Accuracy</div>
              <div style="font-size:48px;font-weight:800;color:#102a43;line-height:1;">{acc:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        class_rows = []
        for class_name in sorted(y.unique()):
            if class_name in report_dict:
                class_rows.append(
                    {
                        "Category": class_name,
                        "Precision": report_dict[class_name]["precision"],
                        "Recall": report_dict[class_name]["recall"],
                        "F1-score": report_dict[class_name]["f1-score"],
                    }
                )

        if class_rows:
            st.markdown("**Precision / Recall / F1-score**")
            prf_df = pd.DataFrame(class_rows)
            st.dataframe(prf_df.round(3), width="stretch", hide_index=True)

        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_fig = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            title="Confusion matrix",
            color_continuous_scale="Blues",
        )
        cm_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f0f4f8",
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
        st.plotly_chart(cm_fig, width="stretch")


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


st.sidebar.title("Navigation")
st.sidebar.caption("")
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

st.title("Dashboard ML")

df_jobs = load_csv(DATASET_JOBS)
df_hr = load_csv(DATASET_HR)

if page.startswith("Vue dynamique"):
    # Vertical cards explaining the three objectives
    def _card_html(title: str, subtitle: str, body_lines: list[str], accent_color: str = "#2563eb"):
        body = "".join(f"<li style=\"margin-bottom:14px;line-height:1.8;font-size:16px\">{line}</li>" for line in body_lines)
        return f"""
        <div style="
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border-left: 6px solid {accent_color};
            border-radius: 16px;
            padding: 36px;
            margin-bottom: 28px;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
            transition: all 0.3s ease;
        ">
          <h3 style="margin: 0 0 8px 0; color: {accent_color}; font-size: 24px; font-weight: 700;">{title}</h3>
          <div style=\"color: #1e40af; margin-bottom: 22px; font-size: 16px; font-weight: 600; letter-spacing: 0.5px;\">{subtitle}</div>
          <ul style=\"margin: 16px 0 0 32px; padding: 0; color: #334155; font-size: 15px; list-style-type: disc;\">
            {body}
          </ul>
        </div>
        """

    # Display three objective cards horizontally
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(_card_html(
            "Extraction automatique des competences",
            "Extraction intelligente du texte",
            [
                "Developper un module NLP pour analyser descriptions",
                "Nettoyage et normalisation du texte",
                "Detection des competences cles mentionnees",
                "Classification: techniques / manageriales / soft skills",
                "Sortie: liste structuree et exploitable",
            ],
            accent_color="#2563eb"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(_card_html(
            "Classification automatique des formations",
            "Categoriser et qualifier les activites",
            [
                "Construire un modele de classification supervisee",
                "Categories: techniques / manageriales / transversales",
                "Vectorisation et selection de features",
                "Evaluation multi-modele avec comparaison",
                "Label standardise pour chaque activite",
            ],
            accent_color="#7c3aed"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(_card_html(
            "Segmentation intelligente des collaborateurs",
            "Regrouper par profils et competences",
            [
                "Regroupement automatique selon parcours",
                "Criteres: competences, niveau, activites",
                "Algorithmes: KMeans, Agglomeratif, PCA",
                "Reduction de dimension et visualisation",
                "Profils semantiques et actionables",
            ],
            accent_color="#059669"
        ), unsafe_allow_html=True)

    st.markdown("---")

    # Show dataset previews vertically
    st.subheader("Datasets utilises")
    
    st.markdown(f"**Dataset 1 — Objectif 1 (`{DATASET_JOBS}`)**")
    if df_jobs is None:
        st.error(f"Fichier introuvable: {DATASET_JOBS}")
    else:
        st.write(f"Taille: {df_jobs.shape[0]} lignes × {df_jobs.shape[1]} colonnes")
        st.dataframe(df_jobs.head(100), width='stretch')

    st.markdown(f"**Dataset classification (Objectif 2) — `{DATASET_CLASSIFICATION}`** — pipeline `{OBJECTIF2_PIPELINE}`")
    r_clf = resolve_dataset_path(DATASET_CLASSIFICATION)
    if not r_clf or not os.path.exists(r_clf):
        st.error(f"Fichier introuvable: {DATASET_CLASSIFICATION}")
    else:
        try:
            prev_clf = pd.read_csv(r_clf, nrows=100, engine="python", on_bad_lines="skip")
            st.write(f"Apercu: {len(prev_clf)} lignes (max 100) × {prev_clf.shape[1]} colonnes")
            st.dataframe(prev_clf, width="stretch")
        except Exception as exc:
            st.warning(f"Apercu classification impossible: {exc}")
    
    st.markdown("**Dataset 3 — RH (Objectif 3)**")
    if df_hr is None:
        st.error(f"Fichier introuvable: {DATASET_HR}")
    else:
        st.write(f"Taille: {df_hr.shape[0]} lignes × {df_hr.shape[1]} colonnes")
        st.dataframe(df_hr.head(100), width='stretch')

    st.markdown("---")

elif page.startswith("Objectif 1"):
    if df_jobs is None:
        st.error(f"Fichier introuvable: {DATASET_JOBS}")
    else:
        render_objectif1(df_jobs)

elif page.startswith("Objectif 2"):
    _clf_path = resolve_dataset_path(DATASET_CLASSIFICATION)
    if not _clf_path or not os.path.exists(_clf_path):
        st.error(f"Fichier introuvable: `{DATASET_CLASSIFICATION}`")
    else:
        render_objectif2()

elif page.startswith("Objectif 3"):
    if df_hr is None:
        st.error(f"Fichier introuvable: {DATASET_HR}")
    else:
        render_objectif3_page(df_hr)