"""Small local matching branch; all geometry/normalization lives in its ONNX graph."""
import torch
from torch import nn
from torch.nn import functional as F


def local_correlation(previous, current):
    """At current (y,x), compare previous (y+dy,x+dx), dy/dx in -1,0,1.

    Explicit slices/exportable arithmetic avoid a custom correlation operator.
    Outside support is zero; edge cells therefore have fewer supported matches.
    """
    previous = previous / torch.sqrt((previous * previous).sum(1, keepdim=True) + 1e-6)
    current = current / torch.sqrt((current * current).sum(1, keepdim=True) + 1e-6)
    height, width = current.shape[-2:]
    padded = F.pad(previous, (1, 1, 1, 1))
    return torch.cat([(current * padded[:, :, 1 + dy:1 + dy + height,
                                        1 + dx:1 + dx + width]).sum(1, keepdim=True)
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dim=1)


class CorrelationMotionNet(nn.Module):
    def __init__(self, base, target_mean, target_std, variant="temporal"):
        super().__init__()
        if variant not in ("temporal", "single-frame"):
            raise ValueError("unknown correlation variant")
        self.variant = variant
        self.base = base.requires_grad_(False)
        self.encoder = nn.Sequential(nn.Conv2d(1, 8, 3, 2, 1), nn.ReLU(),
                                     nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU())
        self.rotation_head = nn.Sequential(nn.Linear(152, 32), nn.ReLU(), nn.Linear(32, 3))
        nn.init.zeros_(self.rotation_head[-1].weight)
        nn.init.zeros_(self.rotation_head[-1].bias)
        self.register_buffer("rotation_mean", torch.as_tensor(target_mean, dtype=torch.float32)[3:].clone())
        self.register_buffer("rotation_std", torch.as_tensor(target_std, dtype=torch.float32)[3:].clone())

    def correction(self, images, state, rotation_prior):
        current = self.encoder(images[:, 1:2])
        # Explicit same-image information control; original translation still sees both frames.
        previous = self.encoder(images[:, :1] if self.variant == "temporal" else images[:, 1:2])
        volume = local_correlation(previous, current)
        spatial = F.avg_pool2d(volume, (3, 4)).flatten(1)
        features = torch.cat((spatial, rotation_prior / .01, state[:, 9:14]), dim=1)
        return self.rotation_head(features) * .01

    def forward(self, images, state, rotation_prior):
        translation = self.base(images, state)[:, :3]
        rotation = (self.correction(images, state, rotation_prior) - self.rotation_mean) / self.rotation_std
        return torch.cat((translation, rotation), dim=1)
