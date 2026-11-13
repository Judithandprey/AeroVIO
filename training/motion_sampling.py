"""Training-only motion strata and deterministic equal-exposure sampling."""
import numpy as np


def classify_motion(times, speed, turn_deg_s, settings):
    """One label per adjacent-frame pair; GT never becomes network input.

    Cruise means approximately constant scalar speed with small angular speed,
    not zero acceleration in all three axes. Startup also contains low-speed
    and accelerating/decelerating transitions, so every pair has one stratum.
    """
    times, speed, turn_deg_s = map(np.asarray, (times, speed, turn_deg_s))
    if len(times) != len(speed) + 1 or speed.shape != turn_deg_s.shape:
        raise ValueError("one speed and turn rate per frame pair required")
    if len(speed) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("need increasing frame times")
    if not all(np.isfinite(v).all() for v in (times, speed, turn_deg_s)):
        raise ValueError("nonfinite motion")
    width = settings['speed_smoothing_frames']
    if width < 1 or width % 2 != 1:
        raise ValueError("smoothing width must be positive and odd")
    smoothed = np.convolve(np.pad(speed, (width//2, width//2), mode='edge'),
                           np.ones(width)/width, mode='valid')
    acceleration = np.gradient(smoothed, times[1:])
    labels = np.ones(len(speed), dtype=np.int64)
    labels[(speed < settings['low_speed_m_s']) |
           (np.abs(acceleration) > settings['tangential_acceleration_m_s2'])] = 0
    labels[turn_deg_s >= settings['turn_deg_s']] = 2
    labels[times[1:] - times[0] <= settings['startup_seconds']] = 0
    return labels, acceleration


def balanced_order(labels, epoch, seed):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (0, 1, 2)).all() or len(labels) % 3:
        raise ValueError("three exhaustive strata and epoch size divisible by three required")
    rng = np.random.default_rng(seed + 1000 + epoch)
    quota = len(labels)//3
    parts = []
    for group in range(3):
        pool = np.flatnonzero(labels == group)
        if not len(pool):
            raise ValueError("empty motion stratum")
        chunks = [rng.permutation(pool) for _ in range((quota + len(pool)-1)//len(pool))]
        parts.append(np.concatenate(chunks)[:quota])
    return rng.permutation(np.concatenate(parts)).astype(np.int64)


def validate_plan(order, epochs, pairs):
    order = np.asarray(order)
    if order.shape != (epochs, pairs) or order.dtype.kind not in 'iu':
        raise ValueError("sampling plan must have one integer index per original epoch example")
    if np.any(order < 0) or np.any(order >= pairs):
        raise ValueError("sampling plan indexes outside the training cache")
    return order.astype(np.int64)
