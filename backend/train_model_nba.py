# backend/train_model_nba.py

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load the team game log
df = pd.read_csv('data/team_1610612747_202324_gamelog.csv', parse_dates=['GAME_DATE'])

# 2. Sort by date
df.sort_values('GAME_DATE', inplace=True)

# 3. Create binary label: 1 if win, 0 if loss
df['W'] = (df['WL'] == 'W').astype(int)

# 4. Rolling averages over last 5 games
df['pts_5']     = df['PTS'].rolling(5).mean()
df['win_pct_5'] = df['W'].rolling(5).mean()

# 5. Drop early rows with NaNs
df = df.dropna(subset=['pts_5', 'win_pct_5'])

# 6. Features & target
FEATURES = ['pts_5', 'win_pct_5']
X = df[FEATURES]
y = df['W']

# 7. Train/test split (80% train, 20% test, time‐ordered)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 8. Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(max_iter=1000))
])

# 9. Train
pipeline.fit(X_train, y_train)

# 10. Evaluate
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]

print("Test Accuracy :", accuracy_score(y_test, y_pred))
print("Test ROC AUC  :", roc_auc_score(y_test, y_proba))

# 11. Save the model
joblib.dump(pipeline, 'model_nba.pkl')
print("✅ Saved trained model to model_nba.pkl")
