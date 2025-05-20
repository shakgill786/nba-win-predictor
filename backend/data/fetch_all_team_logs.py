# backend/data/fetch_all_team_logs.py

import time
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog

def fetch_all_team_logs(season='2024-25'):
    """
    Fetches every NBA team’s game log for the given season
    and writes one CSV per team into backend/data/.
    """
    # 1) Get the list of teams
    all_teams = teams.get_teams()  
    print(f"Found {len(all_teams)} teams. Starting downloads…")

    # 2) Loop & fetch
    for t in all_teams:
        team_id   = t['id']
        abbrev    = t['abbreviation']
        full_name = t['full_name']
        print(f"➡️  Fetching {full_name} ({abbrev})…")

        # 3) Pull the game log
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star='Regular Season'
        )
        df = gamelog.get_data_frames()[0]

        # 4) Save to CSV
        out_path = f"data/{abbrev}_{team_id}_{season.replace('-', '')}_gamelog.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Saved to {out_path}")

        # 5) Be kind to the API
        time.sleep(0.6)  

    print("🎉 All team logs downloaded.")

if __name__ == "__main__":
    # Change season as needed
    fetch_all_team_logs(season='2024-25')
