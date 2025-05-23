# backend/app.py

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# 1️⃣ Load the calibrated ensemble
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_global_ensemble_calibrated.pkl")
model      = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return "🛡️ NBA Win–Predictor API is up! POST JSON to /predict", 200


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({
            "usage": (
                "POST JSON to /predict with keys: "
                "team, pts_5, reb_5, ast_5, win_pct_5, "
                "opp_win_pct_5, fg_pct_5, fg3_pct_5, ft_pct_5, pace_5, "
                "opp_def_rtg, days_rest, back2back, home, opp"
            )
        }), 200

    data = request.get_json(force=True)
    required = [
        "team",
        "pts_5","reb_5","ast_5","win_pct_5",
        "opp_win_pct_5","fg_pct_5","fg3_pct_5","ft_pct_5","pace_5",
        "opp_def_rtg","days_rest","back2back","home","opp"
    ]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify(error=f"Missing required fields: {missing}"), 400

    # 2️⃣ Build a DataFrame with exactly those columns
    X = pd.DataFrame([{k: data[k] for k in required}])
    prob = model.predict_proba(X)[0, 1]
    return jsonify(win_probability=round(float(prob), 4)), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
