import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# 1. Definirea problemei
nb.cells.append(nbf.v4.new_markdown_cell("# 1. Definirea Problemei\n\nAcest notebook abordează problema **predicției părăsirii clienților (Customer Churn)** (Clasificare). \n\n**Obiectiv:** Clasificarea clienților în funcție de probabilitatea de a rezilia contractul cu compania.\n\n**Relevanță practică:** Permite companiilor să identifice clienții la risc și să aplice strategii de retenție.\n\n**Clase:** `Churn` (Yes / No).\n\n**Variabile de intrare:** Gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, etc."))

# 2. Analiza exploratorie a datelor și pregătirea lor
nb.cells.append(nbf.v4.new_markdown_cell("# 2. Analiza Exploratorie a Datelor (EDA) și Pregătirea lor"))
nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Încărcare date
df = pd.read_csv('../data/classification_churn.csv')
# Curățare: TotalCharges are spații goale care trebuie tratate
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df.head()"""))

nb.cells.append(nbf.v4.new_markdown_cell("## Vizualizare\n\nProporția Churn."))
nb.cells.append(nbf.v4.new_code_cell("""sns.countplot(x='Churn', data=df)
plt.title('Distribuția Churn')
plt.show()"""))

nb.cells.append(nbf.v4.new_markdown_cell("## Preprocesare\n\nCodificăm variabilele categorice."))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.preprocessing import LabelEncoder

# Codificare variabila target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Codificare variabile categorice
cat_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop('Churn', axis=1).astype(float)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)"""))

# 3. Antrenarea și compararea modelelor de bază
nb.cells.append(nbf.v4.new_markdown_cell("# 3. Antrenarea și Compararea Modelelor de Bază"))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

models = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0),
    'EBM': ExplainableBoostingClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

df_results = pd.DataFrame(results).sort_values(by='F1', ascending=False)
print(df_results.to_markdown())"""))

# 4. Ajustarea hiperparametrilor (pentru top 5)
nb.cells.append(nbf.v4.new_markdown_cell("# 4. Ajustarea Hiperparametrilor"))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import GridSearchCV

top_5_models = df_results['Model'].head(5).tolist()
print(f'Top 5 modele: {top_5_models}')

# Exemplu pentru Logistic Regression
param_grid = {
    'C': [0.1, 1, 10]
}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)
print(f'Cel mai bun scor Logistic Regression: {grid_search.best_score_}')"""))

# 5. Curbele de învățare
nb.cells.append(nbf.v4.new_markdown_cell("# 5. Curbele de Învățare"))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import learning_curve

def plot_learning_curve(model, name):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=3, scoring='f1')
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation')
    plt.title(f'Learning Curve - {name}')
    plt.legend()
    plt.show()

plot_learning_curve(grid_search.best_estimator_, 'Tuned Logistic Regression')"""))

# 6. Explicabilitatea și analiza SHAP
nb.cells.append(nbf.v4.new_markdown_cell("# 6. Explicabilitatea și Analiza SHAP"))
nb.cells.append(nbf.v4.new_code_cell("""import shap
import warnings
warnings.filterwarnings('ignore') # Ignore SHAP warnings

explainer = shap.Explainer(grid_search.best_estimator_, X_train)

# Handle additivity check safely
try:
    if "check_additivity" in shap.Explainer.__call__.__code__.co_varnames:
         shap_values = explainer(X_test, check_additivity=False)
    else:
         shap_values = explainer(X_test)
except Exception:
    shap_values = explainer(X_test)

# Use bar plot as it is more stable across versions
try:
    shap.summary_plot(shap_values, X_test, plot_type="bar")
except Exception as e:
    print(f"SHAP plotting failed: {e}")
"""))

# Salvare
output_path = 'notebooks/2_classification_churn.ipynb'
with open(output_path, 'w') as f:
    nbf.write(nb, f)
print(f"Notebook salvat in {output_path}")
