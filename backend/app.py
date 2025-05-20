# backend/app.py
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# 1️⃣ Load your global model (re-name or point to model_global_nba.pkl)
model = joblib.load("model_global_nba.pkl")

@app.route("/", methods=["GET"])
def home():
    return "🛡️ NBA Win–Predictor API is up! POST JSON to /predict", 200

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({
            "usage": (
                "POST JSON to /predict with keys: team, pts_5, reb_5, ast_5, "
                "win_pct_5, days_rest, back2back, home, opp"
            )
        }), 200

    # 2️⃣ On POST, fetch and validate JSON
    data = request.get_json(force=True)
    required = ["team","pts_5","reb_5","ast_5","win_pct_5","days_rest","back2back","home","opp"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify(error=f"Missing required fields: {missing}"), 400

    # 3️⃣ Build a DataFrame row and predict
    X = pd.DataFrame([data])
    prob = model.predict_proba(X)[0, 1]

    return jsonify(win_probability=round(float(prob), 4)), 200

if __name__ == "__main__":
    app.run(debug=True)
