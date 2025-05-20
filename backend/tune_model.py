# backend/tune_model.py

import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

# 1) Load your new richer features
df = pd.read_csv("data/all_teams_features_richer_2025.csv", parse_dates=["GAME_DATE"])
df = df.sort_values("GAME_DATE")

# 2) Define X, y
NUMERIC = ["pts_5","reb_5","ast_5","win_pct_5",
           "days_rest","back2back","opp_win_pct_5",
           "fg_pct_5","fg3_pct_5","ft_pct_5","pace_5"]
CATEGORICAL = ["home","opp","team"]
FEATURES = NUMERIC + CATEGORICAL

X = df[FEATURES]
y = df["W"]

# 3) Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
])

# 4) XGBoost pipeline
pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
])

# 5) Hyperparameter distributions
param_dist = {
    "clf__n_estimators": randint(50, 300),
    "clf__max_depth": randint(3, 10),
    "clf__learning_rate": uniform(0.01, 0.3),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.6, 0.4)
}

# 6) TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=25,
    cv=tscv,
    scoring="roc_auc",
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# 7) Fit
search.fit(X, y)

print("Best params:", search.best_params_)
print("Best CV ROC-AUC:", search.best_score_)

# 8) Save best model
joblib.dump(search.best_estimator_, "model_xgb_tuned.pkl")
print("✅ Saved tuned model")
