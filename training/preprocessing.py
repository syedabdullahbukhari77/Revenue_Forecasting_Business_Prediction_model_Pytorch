import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(path: str, training: bool = True, scaler: StandardScaler = None, encoders: dict = None):
    """
    Preprocess the dataset: handle missing values, encode categoricals, scale numerics.
    """
    df = pd.read_csv(path).dropna()

    features = [
        'customer_tenure', 'industry', 'some_other_features',
        'company_size', 'region', 'operating_margin',
        'debt_ratio', 'log_revenue', 'gross_profit'
    ]
    target_cols = ['revenue', 'risk_score', 'churn']

    X = df[features].copy()
    y = df[target_cols] if training else None

    # Encode categoricals
    if training:
        encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
    else:
        for col in X.select_dtypes(include=['object']).columns:
            if col in encoders:
                X[col] = encoders[col].transform(X[col])
            else:
                raise ValueError(f"Encoder for column {col} not provided!")

    # Scale numerics
    if training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    if training:
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y.values, test_size=0.2, random_state=7
        )
        return X_train, X_val, y_train, y_val, scaler, encoders
    else:
        return X_scaled


if __name__ == "__main__":
    # Run preprocessing and save pickles only if executed directly
    X_train, X_val, y_train, y_val, scaler, encoders = load_and_preprocess(
        "synthetic_financial_data_bukharii.csv"
    )
    print("Scaler and encoders saved.")
