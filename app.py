import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

model = joblib.load(MODELS_DIR / "xgb_model.pkl")
feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")

training_data = pd.read_csv(PROCESSED_DIR / "training_data.csv")
teams = sorted(training_data['batting_team'].unique())
venues = sorted(training_data['venue'].unique())

st.title("🏏 IPL Score Predictor")
st.write("Predict the final score of a T20 innings based on the current match situation.")

st.header("Match Situation")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", teams)
    venue = st.selectbox("Venue", venues)
    current_score = st.number_input(
        "Current Score", min_value=0, max_value=300, value=100)

with col2:
    bowling_team = st.selectbox(
        "Bowling Team", [t for t in teams if t != batting_team])
    overs = st.number_input("Overs Completed", min_value=5.0,
                            max_value=20.0, value=10.0, step=0.1)
    wickets = st.number_input(
        "Wickets Fallen", min_value=0, max_value=9, value=2)

if st.button("Predict Final Score"):
    # Build a single row matching our training data structure
    input_dict = {
        'over': overs - 1,  # convert back to zero-indexed, matching training data
        'cumulative_score': current_score,
        'cumulative_wickets': wickets,
        # simplification: treat current score as proxy
        'last_5_overs_runs': current_score,
        'last_5_overs_wickets': wickets,      # simplification: same reasoning
    }

    # Start a row of all zeros for every one-hot column, then set the right ones to 1
    input_row = pd.DataFrame([0] * len(feature_cols), index=feature_cols).T
    for key, value in input_dict.items():
        input_row[key] = value

    batting_col = f"batting_team_{batting_team}"
    bowling_col = f"bowling_team_{bowling_team}"
    venue_col = f"venue_{venue}"

    for col in [batting_col, bowling_col, venue_col]:
        if col in input_row.columns:
            input_row[col] = 1

    prediction = model.predict(input_row)[0]

    st.success(f"Predicted Final Score: **{int(round(prediction))} runs**")
