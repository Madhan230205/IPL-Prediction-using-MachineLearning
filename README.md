# IPL-Prediction-using-MachineLearning

A lightweight Flask web app that serves a pre-trained Scikit-learn win-probability pipeline for IPL matches. Users select teams, toss info, venue, and match state (scores, wickets, run rates) via a simple HTML form; the backend constructs a feature vector, passes it into model_pipeline.pkl, and returns both teams’ win probabilities. This README covers the model’s components, how the input data are prepared, and how to install and run the service locally.
Model Architecture & Training
Pipeline Overview
Type: Scikit-learn sklearn Pipeline combining categorical encoders, feature transformers, and a probabilistic classifier (e.g. GradientBoostingClassifier or RandomForestClassifier).

# Input Features

Categorical: team1, team2, toss_winner, toss_decision, venue

Numerical: score1, wkts1, rr1, (optional) score2, wkts2, rr2

A placeholder toss1_bat column to align with the training schema.

# Training Procedure
Data Source: cleaned_ball_by_ball.csv containing ball-by-ball match states.

# Feature Engineering:

One-hot or ordinal encoding for teams, toss decision, and venue.

Numerical scaling (e.g. MinMaxScaler) for continuous inputs (scores, wickets, run rates).

# Model Selection:

Benchmarked several classifiers via stratified k-fold CV on historical IPL seasons.

Final model chosen based on log-loss and ROC-AUC performance.

# Serialization:

Trained pipeline saved with joblib.dump(pipeline, "model_pipeline.pkl").

# Data Processing Pipeline
CSV Loading

python
Copy
Edit
df = pd.read_csv("cleaned_ball_by_ball.csv")
teams   = sorted(set(df["batting_team"]).union(df["bowling_team"]))
venues  = sorted(df["venue"].unique())
Form Options

Dynamically generate dropdown lists for teams, venues, and toss decisions.

Provide datalists for optional second-innings metrics (score2, wkts2, rr2) including a “Yet to bat” sentinel.

# Request Handling

On POST, retrieve each form field from request.form.

Convert required numeric fields to float; parse optional fields with a helper that maps "Yet to bat" or blank to None.

Feature Assembly

Construct a single-row DataFrame with keys matching the pipeline’s expected feature names.

Call pipeline.predict_proba(X)[0][1] for Team 1 win prob.; Team 2 is 1–p.

# Setup & Installation
Prerequisites
Python 3.8+

# Pip or Poetry for dependency management

Clone & Install
bash
Copy
Edit
git clone https://github.com/Madhan230205/IPL-Prediction-using-MachineLearning.git

# using pip
python -m venv .venv
source .venv/bin/activate        # (Linux/macOS)
.\.venv\Scripts\activate         # (Windows PowerShell)
pip install -r requirements.txt

# or using Poetry
poetry install
requirements.txt
shell
Copy
Edit
Flask>=2.0
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
joblib>=1.1
Model & Data Files
Place model_pipeline.pkl in the project root (exported via joblib).

# Ensure cleaned_ball_by_ball.csv is also in the root (schema as described above).

Running Locally
bash
Copy
Edit
export FLASK_APP=app.py
export FLASK_ENV=development  # enables debug mode
flask run
Visit http://127.0.0.1:5000/ in your browser, fill in match details, and click Predict.
