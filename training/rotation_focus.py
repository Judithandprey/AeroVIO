"""Training-only SO(3) supervision and causal validation for rotation ablations."""
import numpy as np
import torch
from torch.nn import functional as F

from training.motion_model import so3_exp


def geodesic_angle(rotation_a, rotation_b):
    """SO(3) angle in radians, with finite gradients at identity.

    atan2 avoids the acos derivative singularity near a correct prediction.
    The angle remains nondifferentiable at an exact half turn, which is outside
    the small-rotation auxiliary task.
    """
    error = rotation_a.transpose(-1, -2) @ rotation_b
    skew = torch.stack((error[..., 2, 1] - error[..., 1, 2],
                        error[..., 0, 2] - error[..., 2, 0],
                        error[..., 1, 0] - error[..., 0, 1]), -1) * .5
    sine = torch.linalg.vector_norm(skew, dim=-1)
    cosine = (error.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * .5
    return torch.atan2(sine, cosine)


def rotation_auxiliary_loss(raw_motion, state, rotation_gt, scale):
    """Supervise rotation only; auxiliary translations are deliberately unused."""
    if not np.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("rotation loss scale must be finite and positive")
    predicted = so3_exp(state[:, :3]) @ so3_exp(raw_motion[:, 3:])
    normalized = geodesic_angle(predicted, rotation_gt) / scale
    return F.smooth_l1_loss(normalized, torch.zeros_like(normalized))


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _angle_numpy(rotation):
    skew = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]]) * .5
    return float(np.arctan2(np.linalg.norm(skew), (np.trace(rotation) - 1) * .5))


def rollout_metrics(raw_motion, target, state, episode_id):
    """Compose consecutive cached frame pairs exactly as the guest does.

    Each episode starts at identity. Translation uses the previous orientation
    before composition of its relative rotation. RMSE includes the zero-error
    initial frame. No Sim(3), rigid alignment, or future state enters this check.
    """
    raw, target, state = map(_array, (raw_motion, target, state))
    labels = _array(episode_id)
    count = len(raw)
    if count == 0 or raw.shape != (count, 6) or target.shape != (count, 6):
        raise ValueError("rollout needs nonempty Nx6 predictions and targets")
    if state.shape != (count, 14) or labels.shape != (count,):
        raise ValueError("rollout state/episode shape mismatch")
    if not all(np.isfinite(x).all() for x in (raw, target, state, labels)):
        raise ValueError("rollout inputs must be finite")
    if np.any(labels != np.floor(labels)) or np.any(labels < 0):
        raise ValueError("episode_id must contain nonnegative integers")
    with torch.no_grad():
        prior = so3_exp(torch.from_numpy(state[:, :3]))
        predicted = (prior @ so3_exp(torch.from_numpy(raw[:, 3:]))).numpy()
        truth = (prior @ so3_exp(torch.from_numpy(target[:, 3:]))).numpy()
    episodes = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if np.any(np.diff(indices) != 1):
            raise ValueError("episode_id must form contiguous ordered episode blocks")
        rotation_p, rotation_t = np.eye(3), np.eye(3)
        position_p, position_t = np.zeros(3), np.zeros(3)
        rotation_errors, position_errors = [0.0], [0.0]
        for index in indices:
            position_p = position_p + rotation_p @ raw[index, :3]
            position_t = position_t + rotation_t @ target[index, :3]
            rotation_p = rotation_p @ predicted[index]
            rotation_t = rotation_t @ truth[index]
            rotation_errors.append(_angle_numpy(rotation_t.T @ rotation_p))
            position_errors.append(float(np.linalg.norm(position_p - position_t)))
        difference = raw[indices, :3] - target[indices, :3]
        episodes.append({"episode_id": int(label), "pairs": len(indices),
                         "pair_translation_rmse_m": float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1)))),
                         "trajectory_position_rmse_mean_m": float(np.sqrt(np.mean(np.square(position_errors)))),
                         "rotation_rmse_mean_rad": float(np.sqrt(np.mean(np.square(rotation_errors)))),
                         "final_rotation_error_rad": rotation_errors[-1],
                         "final_position_error_m": position_errors[-1]})
    difference = raw[:, :3] - target[:, :3]
    return {"pair_translation_rmse_m": float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1)))),
            "trajectory_position_rmse_mean_m": float(np.mean([e["trajectory_position_rmse_mean_m"] for e in episodes])),
            "rotation_rmse_mean_rad": float(np.mean([e["rotation_rmse_mean_rad"] for e in episodes])),
            "episodes": episodes}


def eligible_checkpoint(metrics, baseline, guard_ratio=1.05):
    if not np.isfinite(guard_ratio) or guard_ratio < 1:
        raise ValueError("translation guard ratio must be finite and at least one")
    keys = ("pair_translation_rmse_m", "trajectory_position_rmse_mean_m", "rotation_rmse_mean_rad")
    if not all(np.isfinite(metrics[key]) and np.isfinite(baseline[key]) for key in keys):
        return False
    # A tiny absolute tolerance prevents rejecting a numerically identical start.
    return all(metrics[key] <= baseline[key] * guard_ratio + 1e-12 for key in keys[:2])
