# backend/dashboard.py

import os
import streamlit as st
import pandas as pd
import joblib

# 1️⃣ Page config (must be the first Streamlit command)
st.set_page_config(page_title="NBA Win Predictor", layout="wide")

# 2️⃣ Load your tuned pipeline locally
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model_xgb_tuned.pkl")
model      = joblib.load(MODEL_PATH)

# 3️⃣ Title
st.title("🏀 NBA Next-Game Win Predictor")

# --- LOAD RICHER FEATURES & TEAM LIST ---
CSV_PATH = os.path.join(os.path.dirname(__file__), "data/all_teams_features_richer_2025.csv")
df       = pd.read_csv(CSV_PATH, parse_dates=["GAME_DATE"])
teams    = sorted(df["team"].unique())

# --- SIDEBAR: SELECT TEAM & INPUTS ---
team    = st.sidebar.selectbox("Select Team", teams)
df_t    = df[df["team"] == team].sort_values("GAME_DATE")
latest  = df_t.iloc[-1]

st.sidebar.header("Latest Rolling Stats")
pts_5        = st.sidebar.slider("5-game avg PTS",         50.0, 150.0, float(latest["pts_5"]))
reb_5        = st.sidebar.slider("5-game avg REB",         20.0,  70.0, float(latest["reb_5"]))
ast_5        = st.sidebar.slider("5-game avg AST",         10.0,  50.0, float(latest["ast_5"]))
win_pct_5    = st.sidebar.slider("5-game win %",           0.0,   1.0, float(latest["win_pct_5"]), step=0.01)
opp_win_pct_5= st.sidebar.slider("Opp 5-game win %",       0.0,   1.0, float(latest["opp_win_pct_5"]), step=0.01)
fg_pct_5     = st.sidebar.slider("5-game FG% avg",         0.0,   1.0, float(latest["fg_pct_5"]), step=0.01)
fg3_pct_5    = st.sidebar.slider("5-game 3P% avg",         0.0,   1.0, float(latest["fg3_pct_5"]), step=0.01)
ft_pct_5     = st.sidebar.slider("5-game FT% avg",         0.0,   1.0, float(latest["ft_pct_5"]), step=0.01)
pace_5       = st.sidebar.slider("5-game Pace (poss/gm)",  80.0, 130.0, float(latest["pace_5"]))

days_rest    = st.sidebar.number_input("Days Rest", 0, 7, int(latest["days_rest"]))
back2back    = st.sidebar.checkbox("Back‐to‐Back?", bool(latest["back2back"]))
home         = st.sidebar.checkbox("Home Game?", bool(latest["home"]))
opp          = st.sidebar.selectbox("Next Opponent", sorted(df_t["opp"].unique()))

# --- PREDICT BUTTON ---
if st.sidebar.button("Predict Next Game"):
    X = pd.DataFrame([{
        "team":           team,
        "pts_5":          pts_5,
        "reb_5":          reb_5,
        "ast_5":          ast_5,
        "win_pct_5":      win_pct_5,
        "opp_win_pct_5":  opp_win_pct_5,
        "fg_pct_5":       fg_pct_5,
        "fg3_pct_5":      fg3_pct_5,
        "ft_pct_5":       ft_pct_5,
        "pace_5":         pace_5,
        "days_rest":      days_rest,
        "back2back":      int(back2back),
        "home":           int(home),
        "opp":            opp,
    }])

    with st.spinner("Calculating…"):
        prob = model.predict_proba(X)[0, 1] * 100
        st.success(f"🏆 Win Probability: {prob:.1f}%")

# --- MAIN CHART: RECENT PERFORMANCE ---
st.header(f"{team} — Recent Performance (Last 20 Games)")
chart_df = df_t.set_index("GAME_DATE")[
    ["pts_5","win_pct_5"]
].tail(20)
st.line_chart(chart_df)

st.markdown(
    """
    *Data & model trained on the 2024-25 season (regular + playoffs) for all teams.*  
    Adjust the sliders to explore different “what-if” scenarios.
    """
)
