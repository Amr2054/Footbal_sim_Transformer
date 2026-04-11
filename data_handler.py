import pickle

import pandas as pd
import numpy as np
import json

def generate_play_tensors(events_filepath, tracking_filepath):
    print("Loading PFF Data...")

    # 1. Load Event Data (JSON)
    with open(events_filepath, 'r') as f:
        events_data = json.load(f)

    events_df = pd.DataFrame(events_data)

    # Group by sequence to find the start of the first event and end of the last event
    sequences = events_df.groupby('sequence').agg(
        start_time=('startTime', 'min'),
        end_time=('endTime', 'max')
    ).reset_index()

    # 2. Load Tracking Data (JSONL)
    print("Loading Tracking Data (JSONL)...")
    tracking_frames = []
    with open(tracking_filepath, 'r') as f:
        for line in f:
            if line.strip():
                tracking_frames.append(json.loads(line))

    tracking_frames = sorted(tracking_frames, key=lambda x: x['videoTimeMs'])

    plays = []
    print(f"Found {len(sequences)} sequences. Extracting plays...")

    # 3. Iterate through sequences to extract plays
    for _, seq_row in sequences.iterrows():
        seq_id = seq_row['sequence']
        start_t = seq_row['start_time']
        end_t = seq_row['end_time']

        start_ms = start_t * 1000
        end_ms = end_t * 1000

        seq_frames = [f for f in tracking_frames if start_ms <= f['videoTimeMs'] <= end_ms]

        if not seq_frames:
            continue

        num_frames = len(seq_frames)

        # 4. Map Players to Fixed Array Indices using JERSEY NUMBERS
        home_ids = set()
        away_ids = set()
        for frame in seq_frames:
            for p in frame.get('homePlayers', []):
                # Use jerseyNum, and ensure it actually exists in this frame
                if p.get('jerseyNum') is not None:
                    home_ids.add(p['jerseyNum'])
            for p in frame.get('awayPlayers', []):
                if p.get('jerseyNum') is not None:
                    away_ids.add(p['jerseyNum'])

        # Limit to 11 players per team
        home_ids = list(home_ids)[:11]
        away_ids = list(away_ids)[:11]

        # 5. Initialize PyTorch-ready Numpy Arrays
        coords = np.zeros((num_frames, 23, 2), dtype=np.float32)
        roles = np.zeros((num_frames, 23), dtype=np.int64)

        roles[:, 0] = 0
        roles[:, 1:12] = 1
        roles[:, 12:23] = 2

        # 6. Populate the Tensors
        for t, frame in enumerate(seq_frames):

            # A. The Ball
            balls = frame.get('balls', [])
            if balls:
                coords[t, 0, 0] = balls[0]['x'] / 52.5
                coords[t, 0, 1] = balls[0]['y'] / 34.0

            # B. Home Players
            for p in frame.get('homePlayers', []):
                pid = p.get('jerseyNum')
                if pid in home_ids:
                    idx = 1 + home_ids.index(pid)
                    coords[t, idx, 0] = p['x'] / 52.5
                    coords[t, idx, 1] = p['y'] / 34.0

            # C. Away Players
            for p in frame.get('awayPlayers', []):
                pid = p.get('jerseyNum')
                if pid in away_ids:
                    idx = 12 + away_ids.index(pid)
                    coords[t, idx, 0] = p['x'] / 52.5
                    coords[t, idx, 1] = p['y'] / 34.0

        # 7. Store the formatted play
        play_data = {
            'sequence_id': int(seq_id),
            'num_frames': num_frames,
            'coordinates': coords,
            'roles': roles
        }
        plays.append(play_data)

    print(f"Successfully extracted {len(plays)} plays!")
    return plays

# --- Example Usage ---
my_plays = generate_play_tensors("FIFA World Cup 2022/Event Data/3812.json",
                                 "FIFA World Cup 2022/Tracking Data/3812.jsonl")
with open("FIFA World Cup 2022/my_plays.pkl", 'wb') as f: # Open in binary write mode ('wb')
            pickle.dump(my_plays, f)

print(my_plays[0]['coordinates'].shape) # Will output something like (105, 23, 2)
