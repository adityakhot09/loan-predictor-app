# train_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_synthetic_data(n=15000):
    age = np.random.randint(18, 75, n)
    income = np.random.normal(40000, 20000, n).clip(8000, 300000)
    employment_years = np.random.poisson(5, n).clip(0, 50)
    existing_debt = np.random.normal(8000, 15000, n).clip(0, 200000)
    num_credit_cards = np.random.poisson(2, n).clip(0, 20)
    late_payments = np.random.poisson(0.4, n).clip(0, 12)
    loan_amount = np.random.normal(15000, 12000, n).clip(500, 200000)
    loan_term_months = np.random.choice([12, 24, 36, 48, 60, 72], n, p=[0.15,0.2,0.25,0.2,0.15,0.05])

    risk_score = (
        (loan_amount / (income + 1)) * 0.6 +
        (existing_debt / (income + 1)) * 0.5 +
        (late_payments * 0.4) +
        (1.0 / (employment_years + 1)) * 5 +
        (num_credit_cards * 0.2)
    )
    default_prob = 1 / (1 + np.exp(-(risk_score - 1.5)))
    labels = (np.random.rand(n) < default_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "employment_years": employment_years,
        "existing_debt": existing_debt,
        "num_credit_cards": num_credit_cards,
        "late_payments": late_payments,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "label_default": labels
    })
    return df

def train_and_save():
    df = generate_synthetic_data(15000)
    X = df.drop(columns=["label_default"])
    y = df["label_default"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_SEED)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print(f"Model accuracy: Train={model.score(X_train, y_train):.3f}, Test={model.score(X_test, y_test):.3f}")

    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("✅ Saved model.joblib and scaler.joblib")

if __name__ == "__main__":
    train_and_save()
