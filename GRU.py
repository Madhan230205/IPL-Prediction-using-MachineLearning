import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. PARAMETERS
JSON_DIR    = 'ipl_male_json'   # folder containing all your .json files
SEQ_LENGTH  = 50                     # number of deliveries per sequence
FEATURES    = ['runs', 'extras', 'wicket']  # features per ball
BATCH_SIZE  = 32
EPOCHS      = 20
OUTPUT_MODEL = 'gru_model.h5'

# 2. FUNCTION to read & flatten one JSON match
def flatten_match(fn):
    with open(fn) as f:
        match = json.load(f)
    rows = []
    mid = match['info'].get('match_id', os.path.basename(fn))
    for inning in match['innings']:
        for over in inning['overs']:
            for d in over['deliveries']:
                rows.append({
                    'match_id': mid,
                    'inning':    inning['team'],
                    'over':      over['over'],
                    'ball':      d.get('ball', np.nan),
                    'runs':      d['runs']['batter'],
                    'extras':    d['runs']['extras'],
                    'wicket':    int('wickets' in d)
                })
    return pd.DataFrame(rows)

# 3. LOAD all matches
all_dfs = []
for fname in os.listdir(JSON_DIR):
    if fname.endswith('.json'):
        df = flatten_match(os.path.join(JSON_DIR, fname))
        all_dfs.append(df)
data = pd.concat(all_dfs, ignore_index=True)
print(f"Total deliveries: {len(data)}")

# 4. BUILD sequences and labels (predict next-ball runs)
X, y = [], []
grouped = data.groupby('match_id')
for _, grp in grouped:
    vals = grp[FEATURES].values
    for i in range(len(vals) - SEQ_LENGTH):
        X.append(vals[i:i+SEQ_LENGTH])
        y.append(vals[i+SEQ_LENGTH][0])
X = np.array(X, dtype='float32')
y = np.array(y, dtype='float32')
print("Sequences:", X.shape, "Labels:", y.shape)

# 5. PAD (should already be uniform, but for safety)
X = pad_sequences(X, maxlen=SEQ_LENGTH, dtype='float32', padding='pre', truncating='pre')

# 6. DEFINE GRU model
model = Sequential([
    GRU(64, input_shape=(SEQ_LENGTH, len(FEATURES)), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 7. TRAIN
model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)

# 8. SAVE to H5
model.save(OUTPUT_MODEL)
print("Saved GRU model to", OUTPUT_MODEL)
