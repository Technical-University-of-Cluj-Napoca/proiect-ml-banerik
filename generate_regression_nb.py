import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# 1. Definirea problemei
nb.cells.append(nbf.v4.new_markdown_cell("# 1. Definirea Problemei\n\nAcest notebook abordează problema **predicției prețului laptopurilor** (Regresie). \n\n**Obiectiv:** Estimarea prețului unui laptop bazat pe specificațiile sale hardware (CPU, RAM, GPU, Greutate, etc.).\n\n**Relevanță practică:** Ajută utilizatorii să determine dacă un laptop este evaluat corect pe piață și oferă producătorilor o metodă de a stabili prețuri competitive.\n\n**Variabila de ieșire:** `Price` (exprimat în unități monetare).\n\n**Variabile de intrare:** Company, TypeName, Inches, ScreenResolution, Cpu, Ram, Memory, Gpu, OpSys, Weight."))

# 2. Analiza exploratorie a datelor și pregătirea lor
nb.cells.append(nbf.v4.new_markdown_cell("# 2. Analiza Exploratorie a Datelor (EDA) și Pregătirea lor"))
nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Încărcare date
df = pd.read_csv('../data/regression_laptops.csv')
if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
    df.drop(df.columns[0], axis=1, inplace=True)
elif df.columns[0] == ',Company':
     df.rename(columns={df.columns[0]: 'ID'}, inplace=True)
     df.drop(df.columns[0], axis=1, inplace=True)

df.head()"""))

nb.cells.append(nbf.v4.new_markdown_cell("## Curățarea datelor\n\nRam și Weight conțin unități de măsură (GB, kg) care trebuie eliminate pentru a deveni numerice."))
nb.cells.append(nbf.v4.new_code_cell("""df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
df.head()"""))

nb.cells.append(nbf.v4.new_markdown_cell("## Vizualizare\n\nDistribuția prețului."))
nb.cells.append(nbf.v4.new_code_cell("""sns.histplot(df['Price'], kde=True)
plt.title('Distribuția Prețului')
plt.show()"""))

nb.cells.append(nbf.v4.new_markdown_cell("## Preprocesare\n\nCodificăm variabilele categorice și împărțim setul de date."))
nb.cells.append(nbf.v4.new_code_cell("""# Extragere tip CPU simplificat (exemplu: Intel Core i5)
df['Cpu_Brand'] = df['Cpu'].apply(lambda x: ' '.join(x.split()[0:3]))

# One-hot encoding pentru variabilele categorice relevante
cat_cols = ['Company', 'TypeName', 'Cpu_Brand', 'OpSys']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Eliminăm coloanele text care nu mai sunt necesare pentru modelul de bază
X = df_encoded.drop(['Price', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu'], axis=1).astype(float)
y = df_encoded['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)"""))

# 3. Antrenarea și compararea modelelor de bază
nb.cells.append(nbf.v4.new_markdown_cell("# 3. Antrenarea și Compararea Modelelor de Bază"))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'Gaussian Process': GaussianProcessRegressor(),
    'XGBoost': XGBRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0),
    'EBM': ExplainableBoostingRegressor()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    })

df_results = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print(df_results.to_markdown())"""))

# 4. Ajustarea hiperparametrilor (pentru top 5)
nb.cells.append(nbf.v4.new_markdown_cell("# 4. Ajustarea Hiperparametrilor"))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import GridSearchCV

top_5_models = df_results['Model'].head(5).tolist()
print(f'Top 5 modele: {top_5_models}')

# Exemplu pentru Random Forest
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)
print(f'Cel mai bun scor RF: {grid_search.best_score_}')"""))

# 5. Curbele de învățare
nb.cells.append(nbf.v4.new_markdown_cell("# 5. Curbele de Învățare"))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import learning_curve

def plot_learning_curve(model, name):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=3, scoring='r2')
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation')
    plt.title(f'Learning Curve - {name}')
    plt.legend()
    plt.show()

plot_learning_curve(grid_search.best_estimator_, 'Tuned Random Forest')"""))

# 6. Explicabilitatea și analiza SHAP
nb.cells.append(nbf.v4.new_markdown_cell("# 6. Explicabilitatea și Analiza SHAP"))
nb.cells.append(nbf.v4.new_code_cell("""import shap
import warnings
warnings.filterwarnings('ignore') # Ignore SHAP warnings

explainer = shap.Explainer(grid_search.best_estimator_, X_train)

# Handle additivity check safely
try:
    # Some explainers (Tree) support check_additivity, others (Linear) do not
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
output_path = 'notebooks/1_regression_laptops.ipynb'
with open(output_path, 'w') as f:
    nbf.write(nb, f)
print(f"Notebook salvat in {output_path}")
