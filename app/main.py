import streamlit as st

st.set_page_config(
    page_title="Proiect Machine Learning",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Analiza Comparată a Modelelor de Machine Learning")

st.markdown("""
Acest proiect prezintă o analiză detaliată a modelelor de Machine Learning pentru două probleme distincte:
1. **Regresie**: Predicția prețului laptopurilor.
2. **Clasificare**: Predicția părăsirii clienților (Customer Churn).

Folosiți bara laterală pentru a naviga între cele două pagini.
""")

st.sidebar.success("Selectați o sarcină de mai sus.")

st.info("Proiect realizat pentru disciplina Sisteme Inteligente.")
