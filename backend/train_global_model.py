import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# 1) Load engineered features
df = pd.read_csv(
    "data/all_teams_features_2025.csv",
    parse_dates=["GAME_DATE"]
)

# 2) Define features & target
NUMERIC     = ["pts_5","reb_5","ast_5","win_pct_5","days_rest","back2back"]
CATEGORICAL = ["home","opp","team"]   # ← use lowercase "team"
FEATURES    = NUMERIC + CATEGORICAL
TARGET      = "W"

X = df[FEATURES]
y = df[TARGET]

# 3) Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
])

# 4) Build pipeline
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression(max_iter=1000))
])

# 5) Time-aware split (80% train / 20% test)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 6) Train
pipeline.fit(X_train, y_train)

# 7) Evaluate
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]
print("Global Accuracy:", accuracy_score(y_test, y_pred))
print("Global ROC AUC :", roc_auc_score(y_test, y_proba))

# 8) Save the model
joblib.dump(pipeline, "model_global_nba.pkl")
print("✅ Saved model_global_nba.pkl")
