"""Explicit causal velocity memory; no true velocity or visual correction input."""
import numpy as np
import torch
from torch import nn

VELOCITY_CLIP = 2.0
MAX_DT = .2


class VelocityMemoryNet(nn.Module):
    def __init__(self, parent=None):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(18, 64), nn.ReLU(), nn.Linear(64, 6))
        if parent is not None:
            with torch.no_grad():
                self.head[0].weight.zero_()
                self.head[0].weight[:, :14].copy_(parent['head.0.weight'])
                self.head[0].bias.copy_(parent['head.0.bias'])
                self.head[2].weight.copy_(parent['head.2.weight'])
                self.head[2].bias.copy_(parent['head.2.bias'])

    def forward(self, inputs):
        return self.head(inputs)


def memory_input(state, memory, valid, mean, std, feedback):
    valid = valid & (state[:, 13] > 0) & (state[:, 13] <= MAX_DT)
    if not feedback:
        valid = torch.zeros_like(valid)
    velocity = torch.where(valid[:, None], memory, torch.zeros_like(memory))
    return torch.cat(((state - mean) / std, velocity, valid[:, None].to(state.dtype)), dim=1)


def advance_memory(raw, rotation, dt, active):
    valid = active & (dt > 0) & (dt <= MAX_DT)
    # delta maps endpoint-camera vectors into start-camera axes.
    memory = torch.bmm(rotation.transpose(1, 2), raw[:, :3, None]).squeeze(-1)
    memory = (memory / dt.clamp_min(1e-6)[:, None]).clamp(-VELOCITY_CLIP, VELOCITY_CLIP)
    return torch.where(valid[:, None], memory, torch.zeros_like(memory)), valid


def sequence_indices(episode_id, targets, length):
    """Left-pad with masked first-frame indices, never cross an episode boundary."""
    ids = np.asarray(episode_id)
    if len(ids) == 0 or length < 1:
        raise ValueError('empty sequence or invalid length')
    starts = np.maximum.accumulate(np.where(np.r_[True, ids[1:] != ids[:-1]], np.arange(len(ids)), 0))
    targets = np.asarray(targets, dtype=np.int64)
    if np.any((targets < 0) | (targets >= len(ids))):
        raise ValueError('target outside cache')
    raw = targets[:, None] + np.arange(1-length, 1)[None]
    active = raw >= starts[targets, None]
    indices = np.maximum(raw, starts[targets, None])
    assert np.all(ids[indices] == ids[targets, None])
    return indices, active
