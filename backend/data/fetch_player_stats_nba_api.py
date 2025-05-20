# backend/data/fetch_player_stats_nba_api.py

import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll

def fetch_player_season_stats(player_id, season):
    """
    Fetch per-game logs for a player in a given season and save to CSV.
    player_id: NBA’s internal player ID (e.g. LeBron James = 2544)
    season:   string in format '2023-24'
    """
    print(f"➡️  Fetching game logs for player {player_id} in season {season}...")
    # Pull the player’s game log
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star='Regular Season'
    )
    df = gamelog.get_data_frames()[0]

    if df.empty:
        print("⚠️  No data returned—check your player_id or season format.")
        return

    out_path = f"data/player_{player_id}_{season.replace('-', '')}_gamelog.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to {out_path}")

if __name__ == "__main__":
    # NBA’s internal LeBron James ID is 2544, season format 'YYYY-YY'
    fetch_player_season_stats(player_id=2544, season='2023-24')
