import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TacticalPlayDataset(Dataset):
    def __init__(self, plays_list, target_frames=250):
        """
        plays_list: The list of dictionaries returned by generate_play_tensors()
        target_frames: The fixed sequence length we want for our Transformer (e.g., 100 frames)
        """
        self.plays = plays_list
        self.target_frames = target_frames

    def __len__(self):
        return len(self.plays)

    def __getitem__(self, idx):
        play = self.plays[idx]
        coords = play['coordinates']  # Shape: (N, 23, 2)
        roles = play['roles']  # Shape: (N, 23)

        current_frames = coords.shape[0]

        # Initialize empty arrays of our target size
        # Coords padded with 0.0, Roles padded with -1 (or a dummy index)
        fixed_coords = np.zeros((self.target_frames, 23, 2), dtype=np.float32)
        fixed_roles = np.zeros((self.target_frames, 23), dtype=np.int64)

        # Create an Attention Mask (1 means real data, 0 means padding)
        # Transformers use this so they don't learn from the padded zeros
        mask = np.zeros((self.target_frames,), dtype=np.float32)

        if current_frames >= self.target_frames:
            # TRUNCATE: Play is too long. Grab the LAST `target_frames` frames.
            fixed_coords = coords[-self.target_frames:, :, :]
            fixed_roles = roles[-self.target_frames:, :]
            mask[:] = 1.0  # All frames are real

        else:
            # PAD: Play is too short. Put the data at the end, leave zeros at the start.
            fixed_coords[-current_frames:, :, :] = coords
            fixed_roles[-current_frames:, :] = roles
            mask[-current_frames:] = 1.0  # Only the end frames are real

        # Convert to PyTorch Tensors
        return {
            'coordinates': torch.tensor(fixed_coords),
            'roles': torch.tensor(fixed_roles),
            'mask': torch.tensor(mask),
            'sequence_id': play['sequence_id']
        }


# --- How to use it ---
# Assuming 'my_plays' is the output from your previous script

import pickle

file_path = "my_plays.pkl"

with open(file_path, 'rb') as f: # Open in binary read mode ('rb')
    my_plays = pickle.load(f)


dataset = TacticalPlayDataset(my_plays, target_frames=100)

# Create a DataLoader to automatically batch your data
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Test it out!
for batch in dataloader:
    print("Batch Coordinates Shape:", batch['coordinates'].shape)
    print("Batch Roles Shape:", batch['roles'].shape)
    break