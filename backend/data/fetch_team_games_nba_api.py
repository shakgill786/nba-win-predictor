# backend/data/fetch_team_games_nba_api.py

import pandas as pd
from nba_api.stats.endpoints import teamgamelog

def fetch_team_season_logs(team_id, season):
    """
    Fetch all games for a team in a given season and save to CSV.
    team_id: NBA’s internal team ID (e.g. Lakers = 1610612747)
    season:  string in format 'YYYY-YY'
    """
    print(f"➡️  Fetching game logs for team {team_id} in season {season}...")
    gamelog = teamgamelog.TeamGameLog(
        team_id=team_id,
        season=season,
        season_type_all_star='Regular Season'
    )
    df = gamelog.get_data_frames()[0]

    if df.empty:
        print("⚠️  No data returned—check your team_id or season format.")
        return

    # Notice we use "team_{team_id}" here
    filename = f"team_{team_id}_{season.replace('-', '')}_gamelog.csv"
    out_path = f"data/{filename}"
    df.to_csv(out_path, index=False)

    print(f"✅ Saved team logs to {out_path}")

if __name__ == "__main__":
    # Lakers = 1610612747, season format 'YYYY-YY'
    fetch_team_season_logs(team_id=1610612747, season='2023-24')
