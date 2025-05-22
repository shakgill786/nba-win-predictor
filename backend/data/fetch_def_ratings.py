# backend/data/fetch_def_ratings.py

import os
import requests
import pandas as pd
from io import StringIO

# Full team name → 3-letter code
LONG_TO_ABBR = {
    "Atlanta Hawks":"ATL","Boston Celtics":"BOS","Brooklyn Nets":"BRK","Charlotte Hornets":"CHO",
    "Chicago Bulls":"CHI","Cleveland Cavaliers":"CLE","Dallas Mavericks":"DAL","Denver Nuggets":"DEN",
    "Detroit Pistons":"DET","Golden State Warriors":"GSW","Houston Rockets":"HOU","Indiana Pacers":"IND",
    "LA Clippers":"LAC","Los Angeles Lakers":"LAL","Memphis Grizzlies":"MEM","Miami Heat":"MIA",
    "Milwaukee Bucks":"MIL","Minnesota Timberwolves":"MIN","New Orleans Pelicans":"NOP","New York Knicks":"NYK",
    "Oklahoma City Thunder":"OKC","Orlando Magic":"ORL","Philadelphia 76ers":"PHI","Phoenix Suns":"PHX",
    "Portland Trail Blazers":"POR","Sacramento Kings":"SAC","San Antonio Spurs":"SAS",
    "Toronto Raptors":"TOR","Utah Jazz":"UTA","Washington Wizards":"WAS"
}

def fetch_def_ratings(season_year: int = 2025):
    """
    1) Download NBA_<year>_ratings.html
    2) Read the first <table> (multi-index columns)
    3) Flatten to single level, extract 'Team' & 'DRtg'
    4) Map to 3-letter codes, write CSV
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_year}_ratings.html"
    print(f"➡️  Downloading {url} …")
    resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
    resp.raise_for_status()

    # read_html will pick up the first table
    dfs = pd.read_html(StringIO(resp.text))
    if not dfs:
        raise RuntimeError("No tables found on ratings page.")
    df = dfs[0]
    print("Original columns:", df.columns.tolist())

    # Flatten multi-index: keep only the second level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1] for col in df.columns]
    print("Flattened columns:", df.columns.tolist())

    # Defensive rating column is 'DRtg' after flattening
    if 'DRtg' not in df.columns or 'Team' not in df.columns:
        raise RuntimeError(f"Couldn't find 'Team' and 'DRtg' in {df.columns.tolist()}")

    # Extract and clean
    out = df[['Team','DRtg']].copy()
    out['Team'] = out['Team'].str.replace(r'\*', '', regex=True)  # drop any asterisks
    out['team'] = out['Team'].map(LONG_TO_ABBR)
    out = out[['team','DRtg']].rename(columns={'DRtg':'opp_def_rtg'})

    # Save CSV
    dest = os.path.join(os.path.dirname(__file__), 'team_def_ratings_2025.csv')
    out.to_csv(dest, index=False)
    print(f"✅ Saved defensive ratings to {dest}")

if __name__ == "__main__":
    fetch_def_ratings(2025)
