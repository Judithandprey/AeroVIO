"""Export the original float32 Gemm/ReLU/Gemm checkpoint as SIMD-friendly arrays."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "experiments/motion-ablation-v2/imu-only-compact"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metadata = json.loads((CHECKPOINT / "meta.json").read_text())
    checkpoint = (CHECKPOINT / "net.onnx").read_bytes()
    if sha256(checkpoint).hexdigest() != metadata["model_sha256"]:
        raise ValueError("ONNX hash mismatch")
    if metadata["network_inputs"] != "state" or metadata["map"]["enabled"]:
        raise ValueError("expected the state-only checkpoint")
    net = cv2.dnn.readNetFromONNX(np.frombuffer(checkpoint, np.uint8))
    layers = [net.getLayer(net.getLayerId(name)) for name in net.getLayerNames()]
    if [layer.type for layer in layers] != ["Gemm", "ReLU", "Gemm"]:
        raise ValueError("expected the original compact ONNX graph")
    lines = ["#pragma once", "#include <array>", f"// ONNX SHA-256 {metadata['model_sha256']}"]

    def literals(row):
        return ", ".join(float(value).hex() + "f" for value in row)

    def vector(name, row):
        lines.append(f"constexpr std::array<float, {len(row)}> {name} = {{{literals(row)}}};")

    for key in ("state_mean", "state_std", "target_mean", "target_std"):
        vector(key, np.asarray(metadata[key], dtype=np.float32))
    for index, (layer, inputs, outputs) in enumerate([(layers[0], 14, 64), (layers[2], 64, 6)], 1):
        weight, bias = layer.blobs
        if weight.shape != (outputs, inputs) or bias.size != outputs or weight.dtype != np.float32 or bias.dtype != np.float32:
            raise ValueError("unexpected Gemm shape or precision")
        padding = -outputs % 4
        weight = np.pad(weight.T, ((0, 0), (0, padding)))
        bias = np.pad(bias.reshape(-1), (0, padding))
        vector(f"bias{index}", bias)
        lines.append(f"constexpr std::array<std::array<float, {len(bias)}>, {inputs}> weight{index} = {{{{")
        lines.extend("    {" + literals(row) + "}," for row in weight)
        lines.append("}};")
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
