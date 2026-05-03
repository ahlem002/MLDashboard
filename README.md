ML Objectives Dashboard
=======================

Small Streamlit dashboard to interact with three ML objectives:

- Objectif 1 — Extraction automatique des compétences
- Objectif 2 — Classification automatique des formations
- Objectif 3 — Segmentation intelligente des collaborateurs (Clustering)

Quickstart
----------

1. Create and activate a Python environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

Files
-----

- [app.py](app.py) — Streamlit application with the three sidebar items.
- [utils.py](utils.py) — Helper functions: extraction, training, prediction, clustering.
- [requirements.txt](requirements.txt) — Python dependencies.

Notes
-----
- The app accepts CSV uploads. For Objective 2 the CSV must contain `text` and `label` columns.
- The app will automatically try to use `JobsDatasetProcessed.csv` and `Cleaned_HR_Data_Analysis.csv` if present in the project root.
- Models are kept only in Streamlit session state; you can extend `utils.py` to save/load pickled models if needed.
