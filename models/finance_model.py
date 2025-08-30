import pandas as pd , seaborn as sns , numpy as np , matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset , DataLoader
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.model_selection import train_test_split

try : 
    df = read_csv('synthetic_financial_data_bukharii.csv')

except FileNotFoundError as e:
    print(f'{e}')

def clean_data():

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df = df.dropna(subset=[col])
            print(f'{col}: {df[col].isnull().sum()}')
    return df

clean_data()

cate_cols : list[str] = ['industry' , 'region' , 'company_size']

target_cols : list[str] = ['revenue' , 'risk_score' , 'churn']

features : list[str] = ['customer_tenure' , 'industry' , 'some_other_features' , 'company_size' , 'region' , 'operating_margin' , 'debt_ratio' , 'log_revenue' , 'gross_profit']

X = df[features].values

y = df[target_cols].values

def encode_data():
    
    encoders = {col: LabelEncoder().fit(df[col]) for col in cate_cols}
    for col in cate_cols:
        df[col] = encoders[col].transform(df[col])
    return df

encode_data()

def scaling_data():
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

y_risk_score , y_revenue_growth , y_churn = df['risk_score'] , df['revenue_growth'] , df['churn']

scaling_data()

X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2 , random_state=42)