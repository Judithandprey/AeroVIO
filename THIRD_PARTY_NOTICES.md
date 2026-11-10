# Upstream attribution

`workspace/solution/solution.py` is the **unchanged KLT + IMU example** from
[jonas-eschmann/viobench](https://github.com/jonas-eschmann/viobench), revision
`2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca`, at
`viobench/solution/examples/python/klt_imu/solution.py`.

The generated `workspace/runtime/`, `workspace/toolchain/`, and workspace ignore
rules also originate from that pinned upstream revision. The upstream project's
`pyproject.toml` declares the MIT license; that checkout has no root LICENSE file.
Original notices in copied source files are preserved. Runtime dependencies
(CPython, NumPy, OpenCV, WASI SDK, Wasmtime, and their dependencies) retain their
respective upstream licenses.

AeroVIO's baseline reproduction and measurement tooling do not claim authorship
of the upstream estimator. Future original changes belong on the experiment
branch, with measured differences attributed separately.

Course PDFs in `docs/course/` retain their authors' rights and are stored in this
private repository for coursework.
