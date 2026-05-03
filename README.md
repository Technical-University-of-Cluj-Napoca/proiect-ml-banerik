# Proiect Machine Learning - Analiza Comparată a Modelelor

Acest proiect realizează o analiză comparată a modelelor de Machine Learning pentru două sarcini fundamentale: Regresie și Clasificare.

## Structura Proiectului

- `data/`: Conține seturile de date utilizate.
- `notebooks/`: Conține caietele Jupyter cu întreg pipeline-ul de analiză.
  - `1_regression_laptops.ipynb`: Predicția prețului laptopurilor.
  - `2_classification_churn.ipynb`: Predicția părăsirii clienților (Churn).
- `app/`: Aplicația Streamlit pentru vizualizarea și testarea modelelor.
- `pyproject.toml`: Gestionarea dependențelor prin `uv`.

## Probleme Abordate

1. **Regresie (Laptop Prices)**: Estimarea prețului laptopurilor bazat pe specificații hardware.
2. **Clasificare (Customer Churn)**: Identificarea clienților care urmează să părăsească serviciile unei companii.

## Modele Incluse

- Linear/Logistic Regression
- Random Forest
- XGBoost
- CatBoost
- Explainable Boosting Machines (EBM)
- SHAP pentru explicabilitate

## Cum se rulează

### 1. Instalarea dependențelor

Proiectul folosește `uv` pentru gestionarea pachetelor.

```bash
uv sync
```

### 2. Rularea Aplicației Streamlit

```bash
uv run streamlit run app/main.py
```

### 3. Vizualizarea Notebook-urilor

```bash
uv run jupyter notebook
```
