# backend/app.py

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# 📦 Load your tuned pipeline (change MODEL_PATH if you used a different name)
MODEL_PATH = os.getenv("MODEL_PATH", "model_xgb_tuned.pkl")
model = joblib.load(MODEL_PATH)


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

    # 1️⃣ Grab the JSON body
    data = request.get_json(force=True)

    # 2️⃣ Make sure nothing’s missing
    required = [
        "team",
        "pts_5", "reb_5", "ast_5", "win_pct_5",
        "opp_win_pct_5", "fg_pct_5", "fg3_pct_5", "ft_pct_5", "pace_5",
        "opp_def_rtg",
        "days_rest", "back2back",
        "home", "opp"
    ]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify(error=f"Missing required fields: {missing}"), 400

    # 3️⃣ Build a single-row DataFrame in the exact order the model expects
    X = pd.DataFrame([{
        "team":            data["team"],
        "pts_5":           data["pts_5"],
        "reb_5":           data["reb_5"],
        "ast_5":           data["ast_5"],
        "win_pct_5":       data["win_pct_5"],
        "opp_win_pct_5":   data["opp_win_pct_5"],
        "fg_pct_5":        data["fg_pct_5"],
        "fg3_pct_5":       data["fg3_pct_5"],
        "ft_pct_5":        data["ft_pct_5"],
        "pace_5":          data["pace_5"],
        "opp_def_rtg":     data["opp_def_rtg"],
        "days_rest":       data["days_rest"],
        "back2back":       data["back2back"],
        "home":            data["home"],
        "opp":             data["opp"],
    }])

    # 4️⃣ Predict and return the probability of a win (class=1)
    prob = model.predict_proba(X)[0, 1]
    return jsonify(win_probability=round(float(prob), 4)), 200


if __name__ == "__main__":
    # listen on 0.0.0.0 for hosting platforms, port picked up from $PORT if set
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
