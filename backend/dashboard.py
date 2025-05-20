# backend/dashboard.py

import streamlit as st
import pandas as pd
import requests

# Page config
st.set_page_config(page_title="NBA Win Predictor", layout="wide")
st.title("🏀 NBA Next-Game Win Predictor")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Input Latest Rolling Stats")
pts_5     = st.sidebar.slider("5-Game Avg Points",    50.0, 140.0, 110.0)
reb_5     = st.sidebar.slider("5-Game Avg Rebounds",  20.0,  60.0,  40.0)
ast_5     = st.sidebar.slider("5-Game Avg Assists",   10.0,  40.0,  25.0)
win_pct_5 = st.sidebar.slider("5-Game Win %",         0.0,   1.0,   0.6, step=0.05)
days_rest = st.sidebar.number_input("Days of Rest",       0,     7,    2)
back2back = st.sidebar.checkbox("Back-to-Back Game?", False)
home      = st.sidebar.checkbox("Home Game?", True)
opp       = st.sidebar.selectbox(
    "Opponent (Abbrev)",
    sorted(["GSW","BKN","MIL","MIA","LAC","PHX","DEN","BOS","NYK","CHI"])
)

# --- PREDICTION BUTTON ---
if st.sidebar.button("Predict Win Probability"):
    payload = {
        "pts_5":      pts_5,
        "reb_5":      reb_5,
        "ast_5":      ast_5,
        "win_pct_5":  win_pct_5,
        "days_rest":  days_rest,
        "back2back":  int(back2back),
        "home":       int(home),
        "opp":        opp
    }
    with st.spinner("Calculating…"):
        try:
            # Hit your Flask API
            res = requests.post("http://127.0.0.1:5000/predict", json=payload)
            res.raise_for_status()
            prob = res.json()["win_probability"] * 100
            st.success(f"🏆 Win Probability: {prob:.1f}%")
        except Exception as e:
            st.error(f"Error fetching prediction: {e}")

# --- MAIN AREA: PERFORMANCE CHART ---
st.header("Recent Team Performance (Last 20 games)")
df = pd.read_csv("data/team_features_2023.csv", parse_dates=["GAME_DATE"])
chart_df = df.set_index("GAME_DATE")[["PTS","pts_5","win_pct_5"]].tail(20)
st.line_chart(chart_df)

st.markdown(
    """
    **How to use this dashboard:**
    1. Adjust your team’s latest rolling stats in the sidebar.  
    2. Click **Predict Win Probability** to get the model’s forecast for the next game.  
    3. See the line chart of points, rolling-5 average, and win% over the last 20 games.
    """
)
