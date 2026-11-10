"""Small paired-image motion network and training-only differentiable warping.

The convolutional backbone follows viobench's attributed tiny_cnn example.
The 14-value state additionally includes camera intrinsics and frame interval.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(64 + 14, 64), nn.ReLU(), nn.Linear(64, 6))

    def forward(self, images, state):
        return self.head(torch.cat((self.features(images), state), dim=1))


def so3_exp(vector):
    """Differentiable Rodrigues formula, finite at zero and small angles."""
    x, y, z = vector.unbind(-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), -1)
    skew = skew.reshape(*vector.shape[:-1], 3, 3)
    theta = torch.linalg.vector_norm(vector, dim=-1)
    a = torch.sinc(theta / torch.pi)
    b = 0.5 * torch.sinc(theta / (2 * torch.pi)).square()
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return identity + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def backward_warp(source, target_depth, intrinsics, motion, imu_rotation):
    """Render target pixels by sampling previous/source image.

    motion[:3] is camera-centre translation in previous camera axes;
    R_prev_current = exp(imu_rotation) @ exp(motion[3:]).
    target_depth is camera-z depth, not ray range. Intrinsics are [fx,fy,cx,cy]
    at the source/target image resolution. Both views have the same calibration.
    """
    batch, _, height, width = source.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=source.device, dtype=source.dtype),
                            torch.arange(width, device=source.device, dtype=source.dtype), indexing="ij")
    fx, fy, cx, cy = [v[:, None, None] for v in intrinsics.unbind(-1)]
    depth = target_depth[:, 0]
    points = torch.stack(((xx - cx) * depth / fx, (yy - cy) * depth / fy, depth), dim=1)
    rotation = so3_exp(imu_rotation) @ so3_exp(motion[:, 3:])
    previous = rotation @ points.reshape(batch, 3, -1) + motion[:, :3, None]
    previous = previous.reshape(batch, 3, height, width)
    z = previous[:, 2].clamp_min(1e-4)
    u = fx * previous[:, 0] / z + cx
    v = fy * previous[:, 1] / z + cy
    grid = torch.stack((2 * (u + 0.5) / width - 1, 2 * (v + 0.5) / height - 1), dim=-1)
    image = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    valid = (previous[:, 2] > 0) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    return image, valid[:, None], previous[:, 2:3], grid


def photometric_loss(images, target_depth, intrinsics, motion, imu_rotation, visibility):
    prediction, _, _, _ = backward_warp(images[:, :1], target_depth, intrinsics, motion, imu_rotation)
    # Visibility comes from true training geometry and is detached/fixed. The
    # network cannot reduce loss by predicting zero confidence or hiding pixels.
    mask = visibility.detach().to(images.dtype)
    error = (prediction - images[:, 1:2]).abs()
    per_sample = (error * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3)).clamp_min(1)
    usable = mask.sum((1, 2, 3)) > 0
    return per_sample[usable].mean() if usable.any() else prediction.sum() * 0
