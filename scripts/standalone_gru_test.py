"""Test: can a 16-unit GRU learn the 3-state HMM when given raw orientation labels?

This isolates V2's learning capacity from V1's noisy signal.
Input: one-hot orientation label (not neural activity)
Output: predict next orientation + classify current state
Training: same HMM sequences as Stage 2

If this gets 95%+ state accuracy → problem is V2's interface with V1
If this gets ~75% → GRU/training setup is the bottleneck
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os

# Use the existing HMM generator
from src.stimulus.sequences import HMMSequenceGenerator

class StandaloneGRU(nn.Module):
    def __init__(self, n_orient=36, hidden_dim=16):
        super().__init__()
        # Input: one-hot orientation (clean, no noise)
        self.gru = nn.GRUCell(n_orient, hidden_dim)
        self.head_state = nn.Linear(hidden_dim, 3)  # CW/CCW/neutral
        self.head_next_ori = nn.Linear(hidden_dim, n_orient)  # next orientation

    def forward(self, orientations_onehot, h=None):
        """
        orientations_onehot: [B, T, N] - one-hot encoded orientations
        Returns: state_logits [B, T, 3], next_ori_logits [B, T, N]
        """
        B, T, N = orientations_onehot.shape
        if h is None:
            h = torch.zeros(B, self.gru.hidden_size, device=orientations_onehot.device)

        state_logits = []
        next_ori_logits = []
        for t in range(T):
            h = self.gru(orientations_onehot[:, t], h)
            state_logits.append(self.head_state(h))
            next_ori_logits.append(self.head_next_ori(h))

        return torch.stack(state_logits, dim=1), torch.stack(next_ori_logits, dim=1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Same params as our model
    gen = HMMSequenceGenerator()

    # Try multiple hidden sizes
    for hidden_dim in [16, 64, 128]:
        print(f"\n=== Hidden dim: {hidden_dim} ===")
        model = StandaloneGRU(n_orient=36, hidden_dim=hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for step in range(2000):
            # Generate batch of HMM sequences
            batch = gen.generate(batch_size=64, seq_length=50)

            # Convert orientations to one-hot (clean labels, no noise)
            ori_channels = (batch.orientations / 5.0).round().long() % 36  # [B, 50]
            ori_onehot = F.one_hot(ori_channels, 36).float().to(device)  # [B, 50, 36]

            # Targets: current state + next orientation
            true_states = batch.states.to(device)  # [B, 50]
            next_ori = torch.roll(ori_channels, -1, dims=1).to(device)  # [B, 50]

            # Forward
            state_logits, next_ori_logits = model(ori_onehot)

            # Loss
            B, T, C = state_logits.shape
            state_loss = F.cross_entropy(state_logits[:, :-1].reshape(-1, 3), true_states[:, :-1].reshape(-1))
            ori_loss = F.cross_entropy(next_ori_logits[:, :-1].reshape(-1, 36), next_ori[:, :-1].reshape(-1))
            loss = state_loss + ori_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                with torch.no_grad():
                    state_acc = (state_logits[:, :-1].argmax(-1) == true_states[:, :-1]).float().mean().item()
                    ori_acc = (next_ori_logits[:, :-1].argmax(-1) == next_ori[:, :-1]).float().mean().item()
                    # Top-3 ori accuracy
                    top3 = next_ori_logits[:, :-1].topk(3, dim=-1).indices
                    top3_acc = (top3 == next_ori[:, :-1].unsqueeze(-1)).any(-1).float().mean().item()
                print(f"  step {step}: loss={loss.item():.3f}, state_acc={state_acc:.3f}, ori_acc={ori_acc:.3f}, top3={top3_acc:.3f}")

if __name__ == '__main__':
    main()
