# 🏏 IPL Score Predictor

Predicts the final score of a T20 IPL innings based on the current match situation, using an XGBoost model trained on historical ball-by-ball data (2008–2024).

## How it works
- Cleaned and merged historical IPL data (fixed team renames like Delhi Daredevils → Delhi Capitals, and venue name inconsistencies like "Eden Gardens" vs "Eden Gardens, Kolkata")
- Engineered features from raw ball-by-ball data: current score, wickets, overs completed, runs in the last 5 overs, batting/bowling team, venue
- Compared a simple run-rate projection baseline against Linear Regression, Random Forest, and XGBoost
- **XGBoost performed best**: ~8.5% average error vs. ~12.8% for the naive run-rate baseline

| Model | MAE | R² | Avg Error |
|---|---|---|---|
| Run-rate baseline | 18.33 | 0.277 | 12.8% |
| Linear Regression | 15.11 | 0.554 | 10.2% |
| Random Forest | 14.03 | 0.602 | 9.3% |
| **XGBoost** | **12.83** | **0.671** | **8.5%** |

## Tech stack
Python, pandas, scikit-learn, XGBoost, Streamlit

## Live demo
https://ipl-score-predictor-mawxaymmo9sgfhkpvsxunr.streamlit.app/

## Dataset
This project uses the [IPL Complete Dataset (2008–2024)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) from Kaggle (`matches.csv` and `deliveries.csv`).

Raw data files are not included in this repo due to size. To reproduce this project from scratch:
1. Download `matches.csv` and `deliveries.csv` from the link above
2. Place them in `data/raw/`
3. Run `src/clean_data.py`, then `src/build_features.py`, then `src/train_model.py`

## Run the app locally
pip install -r requirements.txt
python -m streamlit run app.py


## Limitations
- "Recent momentum" inputs (last-5-overs runs/wickets) are simplified in the live app to current totals, since a single-snapshot dropdown can't capture true recent match momentum
- Trained only on IPL matches; won't generalize to other T20 leagues
- Score prediction is inherently uncertain in T20 cricket — a single big over or late collapse can shift outcomes significantly