"""Shared helpers and fixtures.

Build first (`cmake -S . -B build && cmake --build build`); the tests import
`gb_core` and run `gb_engine` from the build directory, or from $GB_BUILD_DIR.
Every dataset is generated at test time; nothing under tests/ is data.
"""
import contextlib
import io
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("GB_BUILD_DIR", REPO / "build")).resolve()
CLI = BUILD / "gb_engine"
sys.path[:0] = [str(BUILD), str(REPO)]

# GBE1 layout written by GradientBoostingModel::save_model (GradientBoostingModel.hpp:148-170):
# u32 magic, i32 num_features, i32 max_depth, f32 min_child_weight, f32 lambda, f32 gamma,
# f32 learning_rate, f32 base_score, size_t num_trees, then per tree size_t num_nodes + TreeNode[].
HEADER = struct.Struct("<IiifffffQ")
COUNT = struct.Struct("<Q")
# TreeNode (Tree.hpp:17-24) on x86-64 gcc: bool@0, float@4, int@8, uint8@12, int@16, int@20 -> 24 bytes.
NODE = struct.Struct("<?3xfiB3xii")
GBE1_MAGIC = 0x47424531


def parse_model(buf):
    """Return (header tuple, [tree -> [(is_leaf, weight, feature, bin, left, right)]])."""
    header = HEADER.unpack_from(buf, 0)
    assert header[0] == GBE1_MAGIC
    off = HEADER.size
    trees = []
    for _ in range(header[-1]):
        (num_nodes,) = COUNT.unpack_from(buf, off)
        off += COUNT.size
        trees.append([NODE.unpack_from(buf, off + i * NODE.size) for i in range(num_nodes)])
        off += num_nodes * NODE.size
    assert off == len(buf), "model file has trailing or missing bytes"
    return header, trees


def build_model_bytes(num_features, trees, learning_rate=1.0, base_score=0.0, max_depth=6):
    """Serialize a hand-built GBE1 model in the engine's own layout."""
    out = HEADER.pack(GBE1_MAGIC, num_features, max_depth, 1.0, 1.0, 0.0,
                      learning_rate, base_score, len(trees))
    for nodes in trees:
        out += COUNT.pack(len(nodes)) + b"".join(NODE.pack(*n) for n in nodes)
    return out


def run_main(commands, cwd, threads=1):
    """Run main.py in batch mode in a fresh process.

    A fresh process is needed because OpenMP reads OMP_NUM_THREADS when gb_core is
    loaded (main.py:8), so the thread count must be in the environment beforehand.
    """
    env = dict(os.environ, OMP_NUM_THREADS=str(threads),
               PYTHONPATH=os.pathsep.join([str(BUILD), str(REPO)]))
    cmd = [sys.executable, str(REPO / "main.py")]
    for c in commands:
        cmd += ["--cmd", c]
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "[ERROR]" not in proc.stdout, proc.stdout
    return proc.stdout


def final_rmse(stdout):
    return re.search(r"final_rmse=([0-9.]+)", stdout).group(1)


def run_cli(model_path, values):
    return subprocess.run([str(CLI), str(model_path), *values],
                          capture_output=True, text=True, timeout=60)


def write_synthetic(directory, rows, features, seed, qbin=False):
    """Float32 features in [-10, 10] (the engine's fixed range) plus float32 labels."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-10.0, 10.0, size=(rows, features)).astype(np.float32)
    y = (2.0 * X[:, 0] - X[:, 1] + 0.05 * X[:, 2] * X[:, 3]
         + rng.normal(0.0, 0.5, rows)).astype(np.float32)
    directory = Path(directory)
    feat = directory / ("features_qbin.bin" if qbin else "features.bin")
    if qbin:
        # Same mapping as prepare_california_data.py:17-19.
        np.clip((X - np.float32(-10.0)) * np.float32(12.75), 0, 255).astype(np.uint8).tofile(feat)
    else:
        X.tofile(feat)
    labels = directory / "labels.bin"
    y.tofile(labels)
    return feat, labels, X, y


@pytest.fixture(scope="session")
def california_dir(tmp_path_factory):
    """California Housing via scikit-learn, prepared by the repo's own script."""
    pytest.importorskip("sklearn")
    import prepare_california_data

    d = tmp_path_factory.mktemp("california")
    cwd = os.getcwd()
    try:
        os.chdir(d)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_california_data.main()
    finally:
        os.chdir(cwd)
    return d
