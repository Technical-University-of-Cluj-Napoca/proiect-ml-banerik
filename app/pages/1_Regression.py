import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_regression_data, get_regression_model
from sklearn.metrics import mean_absolute_error, r2_score
import shap

st.title("📈 Regresie: Predicția Prețului Laptopurilor")

X, y, df = load_regression_data()

st.header("1. Explorarea Datelor")
st.write("Primele 5 rânduri din setul de date:")
st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.histplot(df['Price'], kde=True, ax=ax)
    ax.set_title("Distribuția Prețului")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(x='Ram', y='Price', data=df, ax=ax)
    ax.set_title("RAM vs Preț")
    st.pyplot(fig)

st.header("2. Testare Modele")
selected_model_name = st.selectbox(
    "Alege un model pentru testare:",
    ['Random Forest', 'XGBoost', 'CatBoost', 'EBM', 'Linear Regression']
)

model = get_regression_model(selected_model_name)
model.fit(X, y)

st.subheader(f"Performanță {selected_model_name}")
y_pred = model.predict(X)
st.write(f"MAE: {mean_absolute_error(y, y_pred):.2f}")
st.write(f"R2 Score: {r2_score(y, y_pred):.4f}")

st.header("3. Predicție Interactivă")
# Create input fields based on X columns
# Simplified version: take common features
st.subheader("Introdu specificațiile laptopului:")
inches = st.slider("Inches", float(df['Inches'].min()), float(df['Inches'].max()), 15.6)
ram = st.number_input("RAM (GB)", int(df['Ram'].min()), int(df['Ram'].max()), 8)
weight = st.number_input("Weight (kg)", float(df['Weight'].min()), float(df['Weight'].max()), 2.0)

# For dummies, we need a more complex way to handle it or just use defaults
# To keep it simple, we'll use a sample row and modify it
input_data = X.iloc[0:1].copy()
input_data.iloc[0] = 0 # reset all to 0
input_data['Inches'] = inches
input_data['Ram'] = ram
input_data['Weight'] = weight

# Optionally let user choose Company/CPU
company = st.selectbox("Company", df['Company'].unique())
if f'Company_{company}' in input_data.columns:
    input_data[f'Company_{company}'] = 1

prediction = model.predict(input_data)[0]
st.success(f"Preț estimat: {prediction:.2f}")

st.header("4. Explicabilitate (SHAP)")
if st.button("Generează SHAP Plot"):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(input_data)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0])
    st.pyplot(fig)
