import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path: str):
    # 1. Load raw data
    df = pd.read_csv(path)

    # 2. Handle missing values
    df = df.dropna()  # or use fillna()

    # 3. Features & target split
    features: list = ['customer_tenure' , 'industry' , 'some_other_features' , 'company_size' , 'region' , 'operating_margin' , 'debt_ratio' , 'log_revenue' , 'gross_profit']
    X = df[features].values
    target_cols : list[str] = ['revenue' , 'risk_score' , 'churn']
    y = df[target_cols].values  # targets

    # 4. Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val, scaler

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, scaler = load_and_preprocess("data/raw/synthetic_financial_data_bukharii.csv")
    print("Preprocessing complete.")
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)
