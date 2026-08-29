import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

matches = pd.read_csv(PROCESSED_DIR / "matches_clean.csv")


def clean_venue(venue):
    # Step 1: strip everything after the first comma (removes city names)
    venue = venue.split(',')[0].strip()

    # Step 2: fix specific known special cases
    manual_fixes = {
        'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
        'Punjab Cricket Association Stadium': 'Punjab Cricket Association IS Bindra Stadium',
        'Sardar Patel Stadium': 'Narendra Modi Stadium',
        'Feroz Shah Kotla': 'Arun Jaitley Stadium',
    }
    return manual_fixes.get(venue, venue)


matches['venue'] = matches['venue'].apply(clean_venue)

print("Unique venues after cleaning:", matches['venue'].nunique())
print(sorted(matches['venue'].unique()))

deliveries = pd.read_csv(PROCESSED_DIR / "deliveries_clean.csv")

deliveries = deliveries[deliveries['match_id'].isin(matches['id'])]

print("Deliveries after filtering:", deliveries.shape)

over_summary = (
    deliveries
    .groupby(['match_id', 'inning', 'over'])
    .agg(runs_this_over=('total_runs', 'sum'),
         wickets_this_over=('is_wicket', 'sum'))
    .reset_index()
    .sort_values(['match_id', 'inning', 'over'])
)

print(over_summary.head(10))

# Running total of runs and wickets as the innings progresses
over_summary['cumulative_score'] = (
    over_summary.groupby(['match_id', 'inning'])['runs_this_over'].cumsum()
)
over_summary['cumulative_wickets'] = (
    over_summary.groupby(['match_id', 'inning'])['wickets_this_over'].cumsum()
)

# The final score of each innings = the last cumulative_score value in that innings
final_scores = (
    over_summary.groupby(['match_id', 'inning'])['cumulative_score']
    .max()
    .reset_index()
    .rename(columns={'cumulative_score': 'final_score'})
)

print(final_scores.head(10))

# Attach the final_score to every over-row of that innings
over_summary = over_summary.merge(final_scores, on=['match_id', 'inning'])

# Rolling sum of the last 5 overs' runs and wickets
over_summary['last_5_overs_runs'] = (
    over_summary.groupby(['match_id', 'inning'])['runs_this_over']
    .transform(lambda x: x.rolling(window=5, min_periods=1).sum())
)
over_summary['last_5_overs_wickets'] = (
    over_summary.groupby(['match_id', 'inning'])['wickets_this_over']
    .transform(lambda x: x.rolling(window=5, min_periods=1).sum())
)

print(over_summary.head(10))

# Get one row per innings with its batting team, bowling team (constant throughout an innings)
team_info = deliveries.groupby(['match_id', 'inning'])[
    ['batting_team', 'bowling_team']].first().reset_index()

# Get venue from matches.csv (note: your column might be 'id' as you found earlier)
venue_info = matches[['id', 'venue']].rename(columns={'id': 'match_id'})

# Merge both into over_summary
over_summary = over_summary.merge(team_info, on=['match_id', 'inning'])
over_summary = over_summary.merge(venue_info, on='match_id')

print(over_summary.head())

training_data = over_summary[over_summary['over'] >= 4].copy()

training_data = training_data[[
    'match_id', 'inning', 'over',
    'batting_team', 'bowling_team', 'venue',
    'cumulative_score', 'cumulative_wickets',
    'last_5_overs_runs', 'last_5_overs_wickets',
    'final_score'
]]

print(training_data.shape)
print(training_data.head())

training_data.to_csv(PROCESSED_DIR / "training_data.csv", index=False)
print("\nSaved training_data.csv")
