# backend/data/fetch_all_team_logs.py

import os
import time
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import TeamGameLog

def fetch_all_team_logs(season='2024-25'):
    # 1) Ensure our data directory exists
    data_dir = os.path.join(os.path.dirname(__file__), '')
    os.makedirs(data_dir, exist_ok=True)

    # 2) Get list of teams
    all_teams = teams.get_teams()
    print(f"Found {len(all_teams)} teams. Downloading logs for season {season}…")

    for t in all_teams:
        team_id   = t['id']
        abbrev    = t['abbreviation']
        full_name = t['full_name']
        print(f"➡️  Fetching {full_name} ({abbrev})…")

        # 3) Fetch Regular Season
        reg = TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]

        # 4) Fetch Playoffs
        po = TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star='Playoffs'
        ).get_data_frames()[0]

        # 5) Combine them
        df = pd.concat([reg, po], ignore_index=True)

        # 6) Save to backend/data/
        out_filename = f"{abbrev}_{team_id}_{season.replace('-','')}_gamelog.csv"
        out_path = os.path.join(data_dir, out_filename)
        df.to_csv(out_path, index=False)
        print(f"✅ Saved {out_path}")

        # 7) Be gentle on the API
        time.sleep(0.6)

    print("🎉 All team logs downloaded.")

if __name__ == "__main__":
    fetch_all_team_logs(season='2024-25')
