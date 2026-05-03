from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=10000)
    X = vect.fit_transform([text])
    arr = X.toarray()[0]
    feats = np.array(vect.get_feature_names_out())
    idx = arr.argsort()[::-1][:top_n]
    return feats[idx].tolist()


def train_text_classifier(df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label'):
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels))>1 else None)
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=20000)
    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    score = clf.score(Xte, y_test)
    return clf, vect, score


def predict_text(text: str, model, vectorizer):
    X = vectorizer.transform([text])
    return model.predict(X)[0]


def cluster_texts(texts: List[str], n_clusters: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=20000)
    X = vect.fit_transform(texts)
    k = KMeans(n_clusters=n_clusters, random_state=42)
    labels = k.fit_predict(X)
    # reduce to 2D for plotting
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X)
    return labels, coords
