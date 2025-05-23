# backend/train_global_ensemble_calibrated.py

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# 1️⃣ Paths
BASE      = os.path.dirname(__file__)
CSV_PATH  = os.path.join(BASE, "data", "all_teams_features_richer_2025.csv")
OUT_MODEL = os.path.join(BASE, "model_global_ensemble_calibrated.pkl")

# 2️⃣ Load & sort
df = pd.read_csv(CSV_PATH, parse_dates=["GAME_DATE"])
df.sort_values(["team", "GAME_DATE"], inplace=True)

# 3️⃣ Features & target
NUMERIC     = [
    "pts_5","reb_5","ast_5","win_pct_5","opp_win_pct_5",
    "fg_pct_5","fg3_pct_5","ft_pct_5","pace_5",
    "days_rest","back2back"
]
CATEGORICAL = ["home","opp","team"]
FEATURES    = NUMERIC + CATEGORICAL
TARGET      = "W"

X = df[FEATURES]
y = df[TARGET]

# 4️⃣ Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
])

# 5️⃣ Base learners & soft-voting ensemble
clf_lr  = LogisticRegression(max_iter=1000, random_state=42)
clf_xgb = XGBClassifier(
    learning_rate=0.01,
    max_depth=3,
    n_estimators=500,
    subsample=1.0,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
clf_rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

ensemble = VotingClassifier(
    estimators=[("lr", clf_lr), ("xgb", clf_xgb), ("rf", clf_rf)],
    voting="soft",
    n_jobs=-1
)

raw_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf",  ensemble)
])

# 6️⃣ Time-series CV
tscv = TimeSeriesSplit(n_splits=5)

# 7️⃣ CV on raw ensemble
roc_auc_raw = cross_val_score(
    raw_pipeline, X, y,
    cv=tscv,
    scoring="roc_auc",
    n_jobs=-1
)
brier_raw = -cross_val_score(
    raw_pipeline, X, y,
    cv=tscv,
    scoring="neg_brier_score",
    n_jobs=-1
)

print("📊 Raw Ensemble CV Metrics (5-fold):")
print(f"  • ROC AUC: {roc_auc_raw.mean():.4f} ± {roc_auc_raw.std():.4f}")
print(f"  • Brier  : {brier_raw.mean():.4f} ± {brier_raw.std():.4f}")

# 8️⃣ Wrap in a CalibratedClassifierCV (sigmoid / Platt), using the same folds
calibrator = CalibratedClassifierCV(
    estimator=raw_pipeline,
    method="sigmoid",
    cv=tscv,
    n_jobs=-1
)

# 9️⃣ CV on calibrated ensemble
roc_auc_cal = cross_val_score(
    calibrator, X, y,
    cv=tscv,
    scoring="roc_auc",
    n_jobs=-1
)
brier_cal = -cross_val_score(
    calibrator, X, y,
    cv=tscv,
    scoring="neg_brier_score",
    n_jobs=-1
)

print("\n📊 Calibrated Ensemble CV Metrics (5-fold):")
print(f"  • ROC AUC: {roc_auc_cal.mean():.4f} ± {roc_auc_cal.std():.4f}")
print(f"  • Brier  : {brier_cal.mean():.4f} ± {brier_cal.std():.4f}")

# 🔟 Finally fit the calibrator on all data
calibrator.fit(X, y)

# 1️⃣1️⃣ Save the calibrated ensemble
joblib.dump(calibrator, OUT_MODEL)
print(f"\n✅ Final calibrated ensemble saved to:\n   {OUT_MODEL}")
