"""Small ONNX motion predictor; no PyTorch or NumPy neural-network fallback."""
import json

import cv2
import numpy as np


class MotionPredictor:
    def __init__(self, bundle):
        self.meta = json.loads(bundle.read_text("meta.json"))
        if self.meta.get("schema_version") != 1:
            raise ValueError("unsupported motion metadata schema")
        self.size = (int(self.meta["input_width"]), int(self.meta["input_height"]))
        if min(self.size) <= 0:
            raise ValueError("network image dimensions must be positive")
        self.state_mean = self._vector("state_mean", 14)
        self.state_std = self._vector("state_std", 14)
        self.target_mean = self._vector("target_mean", 6)
        self.target_std = self._vector("target_std", 6)
        if np.any(self.state_std <= 0) or np.any(self.target_std <= 0):
            raise ValueError("normalization scales must be positive")
        # Missing or incompatible weights must fail visibly, not run another model.
        encoded = np.frombuffer(bundle.read_bytes("net.onnx"), dtype=np.uint8)
        self.net = cv2.dnn.readNetFromONNX(encoded)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net.enableFusion(True)

    def _vector(self, key, length):
        value = np.asarray(self.meta[key], dtype=np.float32)
        if value.shape != (length,) or not np.isfinite(value).all():
            raise ValueError("invalid metadata vector: " + key)
        return value

    def predict(self, pair, state):
        pair = np.asarray(pair, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        if pair.shape != (2, self.size[1], self.size[0]) or state.shape != (14,):
            raise ValueError("incorrect motion network input shape")
        if not np.isfinite(pair).all() or not np.isfinite(state).all():
            raise ValueError("motion network inputs must be finite")
        normalized = (state - self.state_mean) / self.state_std
        self.net.setInput(np.ascontiguousarray(pair[None]), "images")
        self.net.setInput(np.ascontiguousarray(normalized[None]), "state")
        raw = np.asarray(self.net.forward(), dtype=np.float32).reshape(-1)
        if raw.shape != (6,):
            raise ValueError("motion network must return six values")
        motion = raw * self.target_std + self.target_mean
        if not np.isfinite(motion).all():
            raise ValueError("motion network produced non-finite output")
        return motion
