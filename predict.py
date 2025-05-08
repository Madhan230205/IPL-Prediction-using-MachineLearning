from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# --- Load your pipeline once at startup ---
pipeline = joblib.load("model_pipeline.pkl")

# --- Read the CSV and build dropdown lists ---
df = pd.read_csv("cleaned_ball_by_ball.csv")
teams = sorted(set(df["batting_team"]).union(df["bowling_team"].unique()))
venues = sorted(df["venue"].unique())
toss_decisions = ["bat", "field"]

# Prepare options for datalists
score2_opts = ["Yet to bat"] + [str(x) for x in range(0, 301, 10)]
wkts2_opts  = ["Yet to bat"] + [str(x) for x in range(0, 11)]
rr2_opts    = ["Yet to bat"] + [f"{x:.2f}" for x in np.linspace(0, 15, 16)]

HTML = """
<!doctype html>
<title>IPL Win Probability Predictor</title>
<h1>Enter match features</h1>
<form method="post">
  <label>Team 1:
    <select name="team1">
      {% for t in teams %}
        <option value="{{t}}" {% if form.team1==t %}selected{% endif %}>{{t}}</option>
      {% endfor %}
    </select>
  </label><br>

  <label>Team 2:
    <select name="team2">
      {% for t in teams %}
        <option value="{{t}}" {% if form.team2==t %}selected{% endif %}>{{t}}</option>
      {% endfor %}
    </select>
  </label><br>

  <label>Toss Winner:
    <select name="toss_winner">
      {% for t in teams %}
        <option value="{{t}}" {% if form.toss_winner==t %}selected{% endif %}>{{t}}</option>
      {% endfor %}
    </select>
  </label><br>

  <label>Toss Decision:
    <select name="toss_decision">
      {% for d in toss_decisions %}
        <option value="{{d}}" {% if form.toss_decision==d %}selected{% endif %}>{{d}}</option>
      {% endfor %}
    </select>
  </label><br>

  <label>Venue:
    <select name="venue">
      {% for v in venues %}
        <option value="{{v}}" {% if form.venue==v %}selected{% endif %}>{{v}}</option>
      {% endfor %}
    </select>
  </label><br>

  <label>score1:
    <input type="number" name="score1" value="{{ form.score1 or '' }}" required>
  </label><br>

  <label>score2:
    <input list="score2_list" name="score2" value="{{ form.score2 or '' }}">
    <datalist id="score2_list">
      {% for opt in score2_opts %}
        <option value="{{opt}}">
      {% endfor %}
    </datalist>
  </label><br>

  <label>wkts1:
    <input type="number" name="wkts1" value="{{ form.wkts1 or '' }}" required>
  </label><br>

  <label>wkts2:
    <select name="wkts2">
      {% for opt in wkts2_opts %}
        <option value="{{opt}}" {% if form.wkts2==opt %}selected{% endif %}>{{opt}}</option>
      {% endfor %}
    </select>
  </label><br>

  <label>rr1:
    <input type="number" name="rr1" step="0.01" value="{{ form.rr1 or '' }}" required>
  </label><br>

  <label>rr2:
    <input list="rr2_list" name="rr2" value="{{ form.rr2 or '' }}">
    <datalist id="rr2_list">
      {% for opt in rr2_opts %}
        <option value="{{opt}}">
      {% endfor %}
    </datalist>
  </label><br>

  <input type="submit" value="Predict">
</form>

{% if preds %}
  <h2>Results</h2>
  <p>Team 1 win probability: {{ (preds.team1*100)|round(2) }}%</p>
  <p>Team 2 win probability: {{ (preds.team2*100)|round(2) }}%</p>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def predict():
    # default form values
    form = {k: None for k in [
        "team1","team2","toss_winner","toss_decision","venue",
        "score1","score2","wkts1","wkts2","rr1","rr2"
    ]}
    preds = None

    if request.method == "POST":
        # retrieve form inputs
        for k in form:
            form[k] = request.form.get(k)

        # build the input dict with keys matching the pipeline’s features
        data = {
            "team1":         form["team1"],
            "team2":         form["team2"],
            "toss_winner":   form["toss_winner"],
            "toss_decision": form["toss_decision"],
            "venue":         form["venue"],
            "score1":        float(form["score1"]),
            "wkts1":         float(form["wkts1"]),
            "rr1":           float(form["rr1"])
        }

        # helper to parse optional numeric fields
        def parse_optional(val):
            if val in (None, "", "Yet to bat"):
                return None
            return float(val)

        data["score2"]     = parse_optional(form["score2"])
        data["wkts2"]      = parse_optional(form["wkts2"])
        data["rr2"]        = parse_optional(form["rr2"])
        # keep toss1_bat column so pipeline doesn’t break
        data["toss1_bat"]  = None

        X = pd.DataFrame([data])
        p1 = pipeline.predict_proba(X)[0][1]
        preds = {"team1": p1, "team2": 1 - p1}

    return render_template_string(
        HTML,
        teams=teams,
        venues=venues,
        toss_decisions=toss_decisions,
        score2_opts=score2_opts,
        wkts2_opts=wkts2_opts,
        rr2_opts=rr2_opts,
        form=form,
        preds=preds
    )

if __name__ == "__main__":
    app.run(debug=True)
