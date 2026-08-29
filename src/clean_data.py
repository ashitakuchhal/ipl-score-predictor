import pandas as pd
from pathlib import Path

# This finds the project's root folder no matter where the script is run from,
# so file paths always work correctly.
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

matches = pd.read_csv(RAW_DIR / "matches.csv")
deliveries = pd.read_csv(RAW_DIR / "deliveries.csv")

print("Matches shape:", matches.shape)
print("Deliveries shape:", deliveries.shape)

TEAM_RENAMES = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Kings XI Punjab': 'Punjab Kings',
}

# matches.csv has team names in 4 different columns
for col in ['team1', 'team2', 'toss_winner', 'winner']:
    matches[col] = matches[col].replace(TEAM_RENAMES)

# deliveries.csv has team names in 2 columns
for col in ['batting_team', 'bowling_team']:
    deliveries[col] = deliveries[col].replace(TEAM_RENAMES)

all_teams = sorted(set(matches['team1']) | set(matches['team2']))
print("\nFinal team list:", all_teams)
print("Total unique teams:", len(all_teams))

before_count = len(matches)
matches = matches[matches['winner'].notna()]
after_count = len(matches)

print(f"\nDropped {before_count - after_count} matches with no result")
print("Matches remaining:", after_count)

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

matches.to_csv(PROCESSED_DIR / "matches_clean.csv", index=False)
deliveries.to_csv(PROCESSED_DIR / "deliveries_clean.csv", index=False)

print("\nSaved cleaned files to data/processed/")
