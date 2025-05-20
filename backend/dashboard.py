import streamlit as st
import pandas as pd
import requests

# Page config
st.set_page_config(page_title="NBA Win Predictor", layout="wide")
st.title("🏀 NBA Next-Game Win Predictor")

# --- LOAD FEATURES & TEAM LIST ---
df = pd.read_csv("data/all_teams_features_2025.csv", parse_dates=["GAME_DATE"])
teams = sorted(df["team"].unique())

# --- SIDEBAR: SELECT TEAM & INPUTS ---
team = st.sidebar.selectbox("Select Team", teams)
df_t = df[df["team"] == team].sort_values("GAME_DATE")

st.sidebar.header("Latest Rolling Stats")
last = df_t.iloc[-1]
pts_5     = st.sidebar.slider("5-game avg PTS",    50.0, 140.0, float(last["pts_5"]))
reb_5     = st.sidebar.slider("5-game avg REB",    20.0,  60.0,  float(last["reb_5"]))
ast_5     = st.sidebar.slider("5-game avg AST",    10.0,  40.0,  float(last["ast_5"]))
win_pct_5 = st.sidebar.slider("5-game win %",       0.0,   1.0,  float(last["win_pct_5"]), step=0.01)
days_rest = st.sidebar.number_input("Days Rest", 0, 7, int(last["days_rest"]))
back2back = st.sidebar.checkbox("Back-to-Back?", bool(last["back2back"]))
home      = st.sidebar.checkbox("Home Game?", bool(last["home"]))
opp       = st.sidebar.selectbox("Next Opponent", sorted(df_t["opp"].unique()))

# --- PREDICT BUTTON ---
if st.sidebar.button("Predict Next Game"):
    payload = {
        "team":      team,
        "pts_5":     pts_5,
        "reb_5":     reb_5,
        "ast_5":     ast_5,
        "win_pct_5": win_pct_5,
        "days_rest": days_rest,
        "back2back": int(back2back),
        "home":      int(home),
        "opp":       opp
    }
    with st.spinner("Calculating…"):
        res = requests.post("https://<your-api-url>.onrender.com/predict", json=payload)
        res.raise_for_status()
        prob = res.json()["win_probability"] * 100
        st.success(f"🏆 Win Probability: {prob:.1f}%")

# --- MAIN CHART: RECENT PERFORMANCE ---
st.header(f"{team} — Recent Performance (Last 20 Games)")
chart_df = df_t.set_index("GAME_DATE")[["pts_5","win_pct_5"]].tail(20)
st.line_chart(chart_df)

st.markdown(
    """
    *Data & model trained on the 2024-25 season for all teams.*  
    Adjust the slider values or pull in fresh stats via the API.
    """
)
