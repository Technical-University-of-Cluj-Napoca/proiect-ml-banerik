import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier

def load_regression_data():
    df = pd.read_csv('data/regression_laptops.csv')
    if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
        df.drop(df.columns[0], axis=1, inplace=True)
    elif df.columns[0] == ',Company':
         df.drop(df.columns[0], axis=1, inplace=True)
    
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
    df['Cpu_Brand'] = df['Cpu'].apply(lambda x: ' '.join(x.split()[0:3]))
    
    cat_cols = ['Company', 'TypeName', 'Cpu_Brand', 'OpSys']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    X = df_encoded.drop(['Price', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu'], axis=1)
    y = df_encoded['Price']
    return X, y, df

def load_classification_data():
    df = pd.read_csv('data/classification_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    cat_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    return X, y, df

def get_regression_model(name):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'XGBoost': XGBRegressor(),
        'CatBoost': CatBoostRegressor(verbose=0),
        'EBM': ExplainableBoostingRegressor()
    }
    return models.get(name)

def get_classification_model(name):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'XGBoost': XGBClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0),
        'EBM': ExplainableBoostingClassifier()
    }
    return models.get(name)
