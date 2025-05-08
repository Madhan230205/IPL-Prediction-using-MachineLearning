import os
import json
import pandas as pd
import numpy as np

# Directory with your Cricsheet JSON files
JSON_DIR = 'ipl_male_json'

import os
import json
import pandas as pd
import numpy as np

# Directory containing all your ball‑by‑ball JSON files
JSON_DIR = 'ipl_male_json'  # Change this to your directory

def flatten_match(filepath):
    """Flatten one Cricsheet JSON into a DataFrame of deliveries with extra fields."""
    with open(filepath, 'r') as f:
        match = json.load(f)
    info = match.get('info', {})
    
    # Match‑level fields
    match_id     = info.get('match_id', os.path.basename(filepath))
    date         = info.get('dates',[None])[0]
    season       = info.get('season')
    tournament   = info.get('event')
    toss_info    = info.get('toss', {})
    toss_winner  = toss_info.get('winner')
    toss_decision= toss_info.get('decision')
    venue        = info.get('venue')   # stadium name :contentReference[oaicite:0]{index=0}
    city         = info.get('city')    # region name  :contentReference[oaicite:1]{index=1}

    teams = info.get('teams', [])

    rows = []
    for inning in match.get('innings', []):
        batting = inning.get('team')
        bowling = [t for t in teams if t!=batting][0] if len(teams)==2 else None

        for over in inning.get('overs', []):
            over_no = over.get('over')
            # enumerate deliveries to assign ball numbers :contentReference[oaicite:2]{index=2}
            for ball_idx, d in enumerate(over.get('deliveries', []), start=1):
                runs    = d.get('runs',{}).get('batter')
                extras  = d.get('runs',{}).get('extras', 0)
                total   = d.get('runs',{}).get('total', runs+extras)
                wicket  = 1 if d.get('wickets') else 0
                batter  = d.get('batter')
                non_str = d.get('non_striker')
                bowler  = d.get('bowler')

                row = {
                    'match_id':      match_id,
                    'date':          date,
                    'season':        season,
                    'tournament':    tournament,
                    'toss_winner':   toss_winner,
                    'toss_decision': toss_decision,
                    'venue':         venue,
                    'city':          city,
                    'batting_team':  batting,
                    'bowling_team':  bowling,
                    'over':          over_no,
                    'ball':          ball_idx,
                    'batter':        batter,
                    'non_striker':   non_str,
                    'bowler':        bowler,
                    'runs':          runs,
                    'extras':        extras,
                    'runs_total':    total,
                    'wicket':        wicket
                }

                # if wicket details exist, pull kind/player_out/fielders
                wk = d.get('wickets')
                if wk:
                    row['wicket_kind'] = wk[0].get('kind')
                    row['player_out']  = wk[0].get('player_out')
                    for i,f in enumerate(wk[0].get('fielders',[]), start=1):
                        row[f'fielder_{i}'] = f

                rows.append(row)

    return pd.DataFrame(rows)

# 1) Load & flatten all JSON files
all_dfs = []
for fn in os.listdir(JSON_DIR):                                     
    if fn.lower().endswith('.json'):
        df = flatten_match(os.path.join(JSON_DIR, fn))
        all_dfs.append(df)
data = pd.concat(all_dfs, ignore_index=True)                       

print(f"Raw records: {data.shape[0]} deliveries, {data.shape[1]} columns")

# 2) Type conversion & missing‑value handling
data['runs']   = pd.to_numeric(data['runs'],   errors='coerce')    
data['extras'] = pd.to_numeric(data['extras'], errors='coerce')
data = data.dropna(subset=['runs'])                               

# 3) Remove duplicate deliveries
before = data.shape[0]
data = data.drop_duplicates(subset=['match_id','over','ball'])     
print(f"Dropped {before - data.shape[0]} duplicate deliveries")

# 4) Filter out impossible values
data = data[(data['runs']   >= 0) & 
            (data['extras'] >= 0)]                                

# 5) Sort and reset index
data = data.sort_values(['match_id','over','ball']).reset_index(drop=True)

# 6) Export cleaned CSV
OUT_CSV = 'cleaned_ball_by_ball.csv'
data.to_csv(OUT_CSV, index=False)                                  
print(f"Cleaned data saved to {OUT_CSV}, shape: {data.shape}")
