# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Run train_model.py first to create model.joblib and scaler.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def compute_credit_score(features):
    income = max(features.get("income", 0), 1)
    debt = features.get("existing_debt", 0)
    late = features.get("late_payments", 0)
    emp = features.get("employment_years", 0)
    cards = features.get("num_credit_cards", 0)

    debt_to_income = debt / income
    dti_comp = max(0.0, 1.0 - min(debt_to_income, 1.0))
    emp_comp = min(emp / 20, 1.0)
    late_comp = max(0.0, 1.0 - min(late / 6, 1.0))
    card_comp = max(0.0, 1.0 - min(cards / 10, 1.0))
    inc_comp = min(np.log1p(income) / np.log1p(200000), 1.0)

    raw = 0.3*inc_comp + 0.25*dti_comp + 0.2*emp_comp + 0.15*late_comp + 0.1*card_comp
    return int(300 + raw*(850-300))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    name = request.form.get("name", "Applicant")

    def get(name, default=0, type_=float):
        try:
            return type_(request.form.get(name, default))
        except:
            return type_(default)

    features = {
        "age": get("age", 30, int),
        "income": get("income", 30000, float),
        "employment_years": get("employment_years", 2, int),
        "existing_debt": get("existing_debt", 0, float),
        "num_credit_cards": get("num_credit_cards", 1, int),
        "late_payments": get("late_payments", 0, int),
        "loan_amount": get("loan_amount", 5000, float),
        "loan_term_months": get("loan_term_months", 36, int)
    }

    credit_score = compute_credit_score(features)

    X = np.array([[features[k] for k in ["age","income","employment_years","existing_debt","num_credit_cards","late_payments","loan_amount","loan_term_months"]]])
    X_scaled = scaler.transform(X)
    prob_default = model.predict_proba(X_scaled)[0][1]
    prob_pay = 1 - prob_default
    decision = "Yes" if prob_pay >= 0.5 else "No"

    return render_template("index.html", name=name, credit_score=credit_score,
                           decision=decision, prob_pay=f"{prob_pay*100:.1f}%", form_values=features)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
