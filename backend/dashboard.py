# backend/dashboard.py

import os
import streamlit as st
import pandas as pd
import joblib

# 1️⃣ Must be first
st.set_page_config(page_title="NBA Win Predictor", layout="wide")

# 2️⃣ Load your trained pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model_global_nba.pkl")
model      = joblib.load(MODEL_PATH)

# 3️⃣ Title
st.title("🏀 NBA Next-Game Win Predictor")

# 4️⃣ Load your rich features
FEATURES_CSV = os.path.join(
    os.path.dirname(__file__),
    "data/all_teams_features_richer_2025.csv"
)
df = pd.read_csv(FEATURES_CSV, parse_dates=["GAME_DATE"])

# 5️⃣ Load your defensive ratings file
#    (make sure its columns are: team, opp_def_rtg)
DEF_CSV = os.path.join(
    os.path.dirname(__file__),
    "data/team_def_ratings_2025.csv"
)
def_df = (
    pd.read_csv(DEF_CSV)
      .dropna(subset=["team"])          # drop any blank rows
      .assign(team=lambda d: d["team"].str.strip())
)

# 6️⃣ Sidebar: pick your team
teams = sorted(df["team"].unique())
team  = st.sidebar.selectbox("Select Team", teams)
df_t  = df[df["team"] == team].sort_values("GAME_DATE")
latest = df_t.iloc[-1]

# 7️⃣ Sidebar: rolling stats
st.sidebar.header("Latest Rolling Stats")
pts_5         = st.sidebar.slider("5-game avg PTS",         50.0, 150.0, float(latest["pts_5"]))
reb_5         = st.sidebar.slider("5-game avg REB",         20.0,  70.0, float(latest["reb_5"]))
ast_5         = st.sidebar.slider("5-game avg AST",         10.0,  50.0, float(latest["ast_5"]))
win_pct_5     = st.sidebar.slider("5-game win %",           0.0,   1.0,  float(latest["win_pct_5"]), step=0.01)
opp_win_pct_5 = st.sidebar.slider("Opp 5-game win %",       0.0,   1.0,  float(latest["opp_win_pct_5"]), step=0.01)
fg_pct_5      = st.sidebar.slider("5-game FG% avg",         0.0,   1.0,  float(latest["fg_pct_5"]),  step=0.01)
fg3_pct_5     = st.sidebar.slider("5-game 3P% avg",         0.0,   1.0,  float(latest["fg3_pct_5"]), step=0.01)
ft_pct_5      = st.sidebar.slider("5-game FT% avg",         0.0,   1.0,  float(latest["ft_pct_5"]),  step=0.01)
pace_5        = st.sidebar.slider("5-game Pace (poss/gm)",  80.0, 130.0, float(latest["pace_5"]))

# 8️⃣ Rest days slider sized to your data
max_rest     = int(df["days_rest"].max())
default_rest = min(int(latest["days_rest"]), max_rest)
days_rest    = st.sidebar.number_input(
    "Days Rest", 0, max_rest, default_rest
)

back2back = st.sidebar.checkbox("Back-to-Back?", bool(latest["back2back"]))
home      = st.sidebar.checkbox("Home Game?",   bool(latest["home"]))

# 9️⃣ Opponent picker (or free-text fallback)
opponents = sorted(df_t["opp"].unique())
if opponents:
    opp = st.sidebar.selectbox("Next Opponent", opponents)
else:
    opp = st.sidebar.text_input("Next Opponent")

# 🔟 Opponent defensive rating slider
#    default pulled from your `team_def_ratings_2025.csv`
if opp in def_df["team"].values:
    default_def = float(def_df.loc[def_df["team"] == opp, "opp_def_rtg"].iloc[-1])
else:
    default_def = float(def_df["opp_def_rtg"].mean())

min_def = float(def_df["opp_def_rtg"].min())
max_def = float(def_df["opp_def_rtg"].max())

opp_def_rtg = st.sidebar.slider(
    "Opponent Defensive Rating",
    min_value=min_def,
    max_value=max_def,
    value=default_def,
    step=0.1
)

# 1️⃣1️⃣ Make a prediction
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
        "opp_def_rtg":    opp_def_rtg,
    }])

    with st.spinner("Calculating…"):
        prob = model.predict_proba(X)[0, 1] * 100
        st.success(f"🏆 Win Probability: {prob:.1f}%")

# 1️⃣2️⃣ Show recent rolling performance
st.header(f"{team} — Recent Rolling Performance (Last 20 Games)")
chart_df = df_t.set_index("GAME_DATE")[["pts_5","win_pct_5"]].tail(20)
st.line_chart(chart_df)

st.markdown(
    """
    *Data & model trained on the 2024-25 season (regular + playoffs).*  
    Adjust the sliders (including defensive rating) to explore what-ifs.
    """
)
