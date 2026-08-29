# IPL Score Predictor

### Machine Learning-Based T20 Innings Score Prediction

An end-to-end machine learning project that predicts the **final score of an IPL team's innings** based on the current match situation.

The model learns patterns from historical IPL ball-by-ball data and uses information such as the **current score, wickets lost, overs completed, recent scoring patterns, batting team, bowling team, and venue** to estimate how many runs the team is likely to score by the end of the innings.

The trained model is integrated into an interactive **Streamlit web application**, allowing users to enter a match situation and receive a predicted final score.

## 🎯 Project Overview

In a T20 cricket match, the eventual score becomes increasingly predictable as an innings progresses.

For example:

> **CSK is 108/3 after 12 overs. What could their final score be?**

The IPL Score Predictor uses machine learning to estimate the answer based on patterns learned from thousands of historical IPL innings.

The project follows a complete machine learning pipeline:

```text
Historical IPL Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Exploratory Analysis
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Streamlit Application
        ↓
Predicted Final Score
```

## ✨ Key Features

* 🏏 Predicts the final score of an IPL T20 innings
* 📊 Uses historical ball-by-ball IPL data
* 🧹 Cleans and standardizes historical match data
* ⚙️ Performs custom feature engineering
* 📈 Captures current scoring patterns and recent momentum
* 🤖 Compares multiple regression models
* 🏆 Uses XGBoost as the final prediction model
* 🖥️ Provides an interactive Streamlit interface
* 📦 Saves the trained model for inference
* 📊 Evaluates models using multiple regression metrics
* 🚀 Can be deployed as a web application

## 🧠 Machine Learning Problem

This project is formulated as a **supervised regression problem**.

The model receives information about the current state of an innings and predicts a continuous numerical value:

```text
Target (y) = Final Innings Score
```

### Input Features

The model uses match-state information including:

| Feature          | Description                                 |
| ---------------- | ------------------------------------------- |
| `batting_team`   | Team currently batting                      |
| `bowling_team`   | Team currently bowling                      |
| `venue`          | Location of the match                       |
| `overs`          | Overs completed                             |
| `current_score`  | Current innings score                       |
| `wickets`        | Wickets lost                                |
| `last_5_runs`    | Runs scored in the previous five overs      |
| `last_5_wickets` | Wickets lost during the previous five overs |

The final dataset contains:

* **26,702 training rows**
* **73 features**

## 📊 Dataset

The project uses historical **IPL ball-by-ball delivery data**.

Raw delivery data contains individual balls rather than complete match-state snapshots. Therefore, the raw data is transformed into meaningful training examples.

For example:

```text
Batting Team:       CSK
Bowling Team:       MI
Venue:              Wankhede
Overs Completed:    12
Current Score:      108
Wickets Lost:       3
Recent Scoring:     47 runs

                    ↓

            Predicted Final Score
```

Each historical innings can generate multiple snapshots at different stages of the innings.

This allows the model to learn how the state of an innings at a particular point relates to its eventual final score.

## ⚙️ Data Cleaning

Before feature engineering, the historical data is cleaned and standardized.

The preprocessing pipeline includes:

1. Loading match and ball-by-ball datasets
2. Standardizing team names across different IPL seasons
3. Handling historical team-name changes
4. Removing no-result matches
5. Removing delivery records belonging to excluded matches
6. Checking and maintaining consistency between match and delivery data
7. Preparing cleaned datasets for feature engineering

The cleaned data is then used to construct the machine learning dataset.

## 🔧 Feature Engineering

One of the main parts of the project is transforming raw ball-by-ball data into useful machine learning features.

The delivery-level data is first aggregated by:

```text
match → innings → over
```

From this, the following information is calculated:

* Runs scored in each over
* Wickets lost in each over
* Cumulative score
* Cumulative wickets
* Overs completed
* Recent five-over scoring
* Recent five-over wickets
* Batting team
* Bowling team
* Venue
* Final innings score

For example, multiple individual delivery records:

```text
Ball 1 → 1 run
Ball 2 → 4 runs
Ball 3 → 0 runs
Ball 4 → 2 runs
...
```

are converted into a match-state representation such as:

```text
Over:             12
Current Score:    108
Wickets:          3
Last 5 Runs:      47
Final Score:      184
```

This process transforms raw cricket data into a structured dataset suitable for machine learning.

## 🤖 Models Evaluated

Four approaches were evaluated.

### 1. Run-Rate Baseline

A simple run-rate projection was used as a baseline.

For example:

```text
Current Score = 100
Overs Completed = 10

Current Run Rate = 100 / 10 = 10

Projected Final Score = 10 × 20 = 200
```

This provides a benchmark to determine whether machine learning actually improves prediction accuracy.

### 2. Linear Regression

A traditional regression model used as a simple machine learning benchmark.

### 3. Random Forest

A tree-based ensemble model capable of capturing nonlinear relationships between match conditions and final scores.

### 4. XGBoost

XGBoost was used as the final model because it performed best among the evaluated approaches.

It is particularly effective for structured/tabular datasets and can capture complex relationships between the different match-state features.

## 📈 Model Evaluation & Results

The models were evaluated using:

* **MAE — Mean Absolute Error**
* **RMSE — Root Mean Squared Error**
* **R² — Coefficient of Determination**
* **Average Percentage Error**

### Final Results

| Model             |     MAE ↓ |    RMSE ↓ |      R² ↑ | Avg. Error ↓ |
| ----------------- | --------: | --------: | --------: | -----------: |
| Run-Rate Baseline |     18.33 |     26.12 |     0.277 |        12.8% |
| Linear Regression |     15.11 |     20.52 |     0.554 |        10.2% |
| Random Forest     |     14.03 |     19.38 |     0.602 |         9.3% |
| **XGBoost**       | **12.83** | **17.63** | **0.671** |     **8.5%** |

### 🏆 Best Model: XGBoost

XGBoost achieved the best performance across all evaluated metrics.

Compared with the simple run-rate baseline:

* **MAE:** 18.33 → **12.83**
* **RMSE:** 26.12 → **17.63**
* **R²:** 0.277 → **0.671**
* **Average Error:** 12.8% → **8.5%**

This represents approximately a **30% reduction in MAE** compared with the baseline.

An MAE of **12.83** means that, on average, the model's prediction differs from the actual final score by approximately **13 runs**.

The R² score of **0.671** indicates that the model explains approximately **67.1% of the variation** in final scores within the evaluated dataset.

The trained model and required feature columns are saved in the `models/` directory for use by the application.

## 🖥️ Streamlit Application

The trained XGBoost model is integrated into an interactive Streamlit web application.

Users can enter the current state of an IPL innings and request a prediction.

### Example

```text
Batting Team:      CSK
Bowling Team:      MI
Venue:             Wankhede

Current Score:     108
Overs Completed:   12
Wickets Lost:      3

              ↓

       ML Prediction

              ↓

      Predicted Score
             184
```

The application loads the previously trained model instead of retraining it whenever a prediction is requested.

## 🏗️ System Architecture

```text
                HISTORICAL IPL DATA
                         │
                         ▼
                ┌─────────────────┐
                │  Data Cleaning  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Feature      │
                │   Engineering   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Training Dataset│
                │ 26,702 rows     │
                │ 73 features     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Model Training  │
                │ & Evaluation    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     XGBoost     │
                │  Score Predictor│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Streamlit    │
                │      App        │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Predicted Final │
                │      Score      │
                └─────────────────┘

## 📁 Project Structure

```text
ipl-score-predictor/
│
├── data/
│   ├── raw/
│   │   ├── matches.csv
│   │   └── deliveries.csv
│   │
│   └── processed/
│       ├── matches_clean.csv
│       ├── deliveries_clean.csv
│       └── training_data.csv
│
├── models/
│   ├── score_predictor.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── clean_data.py
│   ├── build_features.py
│   └── train.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

### Programming

* Python

### Data Processing

* pandas
* NumPy

### Machine Learning

* scikit-learn
* XGBoost

### Model Persistence

* joblib

### Application

* Streamlit

### Version Control & Deployment

* Git
* GitHub
* Streamlit Community Cloud

---

## 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ipl-score-predictor.git
cd ipl-score-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🌐 Live Demo

🚀 **Live application:** *Add your deployed Streamlit URL here*

## ⚠️ Limitations

Although the model performs well on the historical dataset, cricket is inherently unpredictable and several real-world factors cannot be completely captured by the current model.

Some limitations include:

* Predictions are based primarily on historical IPL data.
* Player-specific form and individual player matchups may not be fully represented.
* Pitch conditions can vary significantly between matches.
* Weather conditions and other environmental factors are not fully incorporated.
* Sudden changes in momentum can be difficult to predict.
* The live application uses a simplified representation of recent match events compared with complete ball-by-ball input.
* The model is trained specifically on IPL matches and may not generalize directly to other T20 leagues.

Therefore, predictions should be interpreted as **statistical estimates rather than guaranteed outcomes**.

## 🔮 Future Improvements

The project can be extended in several directions:

### 🏆 Win Probability Prediction

Develop a separate classification model to estimate the probability of each team winning during a match.

### 🔮 What-If Simulation

Allow users to explore hypothetical situations:

```text
Current prediction: 184

What if the next over produces 15 runs?

        ↓

New prediction: 192

### 👤 Player-Level Features

Incorporate information such as:

* Batter strike rates
* Bowler economy rates
* Player form
* Batter vs. bowler statistics
* Player-specific historical performance

### 🌦️ Match Conditions

Include additional information such as:

* Weather
* Pitch characteristics
* Toss result
* Batting first/second
* Ground dimensions

### 📊 Explainable AI

Use tools such as SHAP to show which features contributed most to an individual prediction.

### 📡 Live Prediction

Integrate live match data to continuously update the predicted final score after each over or delivery.

## 📚 Learning Outcomes

This project provided practical experience with the complete machine learning workflow:

* Working with real-world sports data
* Data cleaning and preprocessing
* Handling inconsistent historical data
* Feature engineering from raw ball-by-ball data
* Exploratory data analysis
* Supervised regression
* Comparing multiple ML algorithms
* Model evaluation using MAE, RMSE, and R²
* XGBoost
* Model persistence using joblib
* Building an ML-powered web application with Streamlit
* Version control with Git and GitHub
* Deploying a machine learning application

## 👩‍💻 Author

**Ashita Kuchhal**

An end-to-end machine learning project exploring the application of predictive modeling to cricket analytics.

## ⭐ Acknowledgements

This project was developed as a practical exploration of machine learning, data preprocessing, feature engineering, and sports analytics using historical IPL data.

## 📌 Disclaimer

This project is intended for **educational and analytical purposes**.

The predictions generated by the model are statistical estimates based on historical data and should not be considered guaranteed match outcomes.
