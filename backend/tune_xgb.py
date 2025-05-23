# backend/tune_xgb.py
import os
import pandas as pd

from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost                 import XGBClassifier

# ── 1) Locate your richer features file
BASE     = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "data", "all_teams_features_richer_2025.csv")

# ── 2) Load & sort chronologically
df = pd.read_csv(CSV_PATH, parse_dates=["GAME_DATE"])
df.sort_values(["team", "GAME_DATE"], inplace=True)

# ── 3) Define your features & target
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

# ── 4) Build the same preprocessor you use in training
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    # in recent sklearn versions `sparse` was renamed to `sparse_output`
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
])

# ── 5) Create the full pipeline
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf",  XGBClassifier(
                 eval_metric="logloss",
                 random_state=42
             ))
])

# ── 6) Time‐series CV splitter
tscv = TimeSeriesSplit(n_splits=5)

# ── 7) Hyperparameter grid
param_grid = {
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__max_depth":      [3, 5, 7],
    "clf__n_estimators":   [100, 300, 500],
    "clf__subsample":      [0.6, 0.8, 1.0],
}

# ── 8) GridSearchCV using built‐in "roc_auc" scorer
gscv = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring="roc_auc",      # just use the string key
    n_jobs=-1,
    verbose=2,
    error_score="raise"
)

print("🔍 Running hyperparameter search…")
gscv.fit(X, y)

print("\n🏆 Best params:", gscv.best_params_)
print("📈 Best CV ROC AUC:", gscv.best_score_)
