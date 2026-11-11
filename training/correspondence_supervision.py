"""Training-only geometric soft correspondence labels for the unchanged v5 grid."""
import torch
from torch.nn import functional as F
from training.motion_model import so3_exp
from training.feature_correlation import local_correlation


def correspondence_targets(depth, intrinsics, motion, raw_imu_rotation, visibility):
    """Map current feature centres to previous image using labeled camera geometry.

    Two k3/s2/p1 convolutions have image-coordinate centres (4*x,4*y),
    not the centre convention of resize-to-16x12. Labels use that exact lattice.
    motion is the original cache's translation and RAW-IMU rotation residual.
    Depth is current camera-z; visibility is a fixed GT-derived mask.
    """
    count, _, height, width = depth.shape
    if (height, width) != (48, 64): raise ValueError("expected 64x48 depth")
    dtype, device = depth.dtype, depth.device
    yy, xx = torch.meshgrid(torch.arange(0, height, 4, dtype=dtype, device=device),
                            torch.arange(0, width, 4, dtype=dtype, device=device), indexing="ij")
    z = depth[:, 0, ::4, ::4]
    fx, fy, cx, cy = (v[:, None, None] for v in intrinsics.unbind(-1))
    points = torch.stack(((xx-cx)*z/fx, (yy-cy)*z/fy, z), 1)
    rotation = so3_exp(raw_imu_rotation) @ so3_exp(motion[:, 3:])
    previous = (rotation @ points.reshape(count, 3, -1) + motion[:, :3, None]).reshape(count, 3, 12, 16)
    projected_z = previous[:, 2]
    safe_z = projected_z.clamp_min(1e-6)
    u, v = fx*previous[:, 0]/safe_z+cx, fy*previous[:, 1]/safe_z+cy
    dx, dy = (u-xx)/4, (v-yy)/4
    valid = visibility[:, 0, ::4, ::4].bool() & (z > 0) & (projected_z > 0)
    valid &= torch.isfinite(dx) & torch.isfinite(dy) & (dx.abs() <= 1) & (dy.abs() <= 1)
    # Every candidate must exist; this also avoids a border-support shortcut.
    interior = (xx >= 4) & (xx <= 56) & (yy >= 4) & (yy <= 40)
    valid &= interior
    dx = torch.where(valid, dx, torch.zeros_like(dx))
    dy = torch.where(valid, dy, torch.zeros_like(dy))
    distribution = torch.stack([((1-(dx-ox).abs()).clamp_min(0) *
                                  (1-(dy-oy).abs()).clamp_min(0))
                                 for oy in (-1, 0, 1) for ox in (-1, 0, 1)], 1)
    distribution = distribution * valid[:, None]
    return {"soft_target": distribution, "valid": valid,
            "displacement": torch.stack((dx, dy), 1)}


def volume(net, images):
    previous = images[:, :1] if net.variant == "temporal" else images[:, 1:2]
    return local_correlation(net.encoder(previous), net.encoder(images[:, 1:2]))


def correspondence_loss(correlation, target, valid, temperature=.1):
    """Equal usable-pair cross entropy, including subpixel bilinear soft labels."""
    loss = -(target * F.log_softmax(correlation/temperature, dim=1)).sum(1)
    counts = valid.sum((1, 2))
    per_pair = (loss*valid).sum((1, 2)) / counts.clamp_min(1)
    if (counts > 0).any(): return per_pair[counts > 0].mean()
    return correlation.sum()*0


def matching_displacement(correlation, temperature=.1):
    probabilities = F.softmax(correlation/temperature, dim=1)
    offsets = correlation.new_tensor([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    return torch.einsum("nkhw,kd->ndhw", probabilities, offsets)
