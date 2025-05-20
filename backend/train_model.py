import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load & sort
df = pd.read_csv("data/team_1610612747_202324_gamelog.csv",
                 parse_dates=["GAME_DATE"],
                 infer_datetime_format=True)
df.sort_values("GAME_DATE", inplace=True)

# 2. Binary win label
df["W"] = (df["WL"] == "W").astype(int)

# 3. Rolling stats
df["pts_5"]     = df["PTS"].rolling(5).mean()
df["reb_5"]     = df["REB"].rolling(5).mean()
df["ast_5"]     = df["AST"].rolling(5).mean()
df["win_pct_5"] = df["W"].rolling(5).mean()

# 4. Rest/back-to-back
df["days_rest"] = df["GAME_DATE"].diff().dt.days.fillna(0)
df["back2back"] = (df["days_rest"] == 1).astype(int)

# 5. Home/away & opponent flags
df["home"] = df["MATCHUP"].str.contains(" vs. ").astype(int)
df["opp"]  = df["MATCHUP"].str.split().str[-1]

# 6. Drop rows with NaNs from rolling windows
df = df.dropna(subset=["pts_5","reb_5","ast_5","win_pct_5","days_rest"])

# 7. Define features & target
NUMERIC     = ["pts_5","reb_5","ast_5","win_pct_5","days_rest","back2back"]
CATEGORICAL = ["home","opp"]
FEATURES    = NUMERIC + CATEGORICAL
TARGET      = "W"

X = df[FEATURES]
y = df[TARGET]

# 8. Preprocessing pipelines
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
])

# 9. Full pipeline
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression(max_iter=1000))
])

# 10. Time-aware split (80% train / 20% test)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 11. Train & evaluate
pipeline.fit(X_train, y_train)
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, y_proba))

# 12. Save the model
joblib.dump(pipeline, "model_nba.pkl")
print("✅ model_nba.pkl saved")
