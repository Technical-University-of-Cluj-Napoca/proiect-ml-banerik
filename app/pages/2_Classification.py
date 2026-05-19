import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_classification_data, get_classification_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import shap

st.title("🎯 Clasificare: Customer Churn Prediction")

X, y, df = load_classification_data()

st.header("1. Explorarea Datelor")
st.write("Primele 5 rânduri din setul de date:")
st.dataframe(df.head())

fig, ax = plt.subplots()
sns.countplot(x='Churn', data=df, ax=ax)
ax.set_title("Distribuția Churn")
st.pyplot(fig)

st.header("2. Testare Modele")
selected_model_name = st.selectbox(
    "Alege un model pentru testare:",
    ['Random Forest', 'XGBoost', 'CatBoost', 'EBM', 'Logistic Regression']
)

model = get_classification_model(selected_model_name)
model.fit(X, y)

st.subheader(f"Performanță {selected_model_name}")
y_pred = model.predict(X)
st.write(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
st.write(f"F1 Score: {f1_score(y, y_pred):.4f}")

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

st.header("3. Predicție Interactivă")
st.subheader("Introdu datele clientului:")
tenure = st.slider("Tenure (luni)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 600.0)

# Simplificăm input-ul pentru demo
input_data = X.iloc[0:1].copy()
input_data.iloc[0] = 0
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges
input_data['TotalCharges'] = total_charges

# Predict
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]

if prediction == 1:
    st.error(f"Clientul va părăsi compania (Probabilitate: {prob:.2f})")
else:
    st.success(f"Clientul va rămâne (Probabilitate Churn: {prob:.2f})")

st.header("4. Explicabilitate (SHAP)")
if st.button("Generează SHAP Plot"):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(input_data)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0])
    st.pyplot(fig)
