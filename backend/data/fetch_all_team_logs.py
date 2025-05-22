# backend/data/fetch_all_team_logs.py

import os
import time
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import TeamGameLog

def fetch_all_team_logs(season="2024-25"):
    # 1️⃣ target dir is backend/data, since this script lives in backend/data/
    data_dir = os.path.dirname(__file__)  # THIS is already .../backend/data
    os.makedirs(data_dir, exist_ok=True)

    # 2️⃣ grab every team
    all_teams = teams.get_teams()
    print(f"Found {len(all_teams)} teams. Downloading logs for season {season}…")

    combined = []
    for t in all_teams:
        tid    = t["id"]
        abbr   = t["abbreviation"]
        name   = t["full_name"]
        print(f"➡️  Fetching {name} ({abbr})…")

        # 3️⃣ regular
        reg_df = TeamGameLog(
            team_id=tid,
            season=season,
            season_type_all_star="Regular Season"
        ).get_data_frames()[0]

        # 4️⃣ playoffs
        po_df  = TeamGameLog(
            team_id=tid,
            season=season,
            season_type_all_star="Playoffs"
        ).get_data_frames()[0]

        # 5️⃣ drop any columns that are 100% NA in each, to avoid pandas warning
        reg_df = reg_df.dropna(axis=1, how="all")
        po_df  = po_df.dropna(axis=1, how="all")

        # 6️⃣ concat (sort=False preserves column order)
        df = pd.concat([reg_df, po_df], ignore_index=True, sort=False)

        # 7️⃣ add a team-abbrev column
        df["team"] = abbr
        combined.append(df)

        time.sleep(0.6)  # gentle

    # 8️⃣ merge all teams
    all_games = pd.concat(combined, ignore_index=True, sort=False)

    # 9️⃣ save one big CSV
    out_path = os.path.join(data_dir, "all_teams_202425_gamelog.csv")
    all_games.to_csv(out_path, index=False)
    print(f"✅ Saved combined logs to {out_path}")

if __name__ == "__main__":
    fetch_all_team_logs()
