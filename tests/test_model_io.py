"""Model persistence and agreement between the Python and native inference paths."""
import contextlib
import io

import numpy as np
import pytest

import gb_core
import main
from conftest import build_model_bytes, run_cli, write_synthetic

ROWS, FEATURES = 3_000, 6


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A small ensemble trained in-process through the shell's own training loop."""
    d = tmp_path_factory.mktemp("io")
    feat, labels, X, _ = write_synthetic(d, ROWS, FEATURES, seed=3)
    shell = main.GBEngineShell()
    with contextlib.redirect_stdout(io.StringIO()):
        shell.onecmd(f"load_train {feat} {labels} {FEATURES}")
        shell.onecmd("train 8 4 0.3 1.0")
    assert shell.model is not None and shell.model.get_num_trees() == 8
    return d, shell, X


def _bins(X):
    return np.clip((X - np.float32(-10.0)) * np.float32(12.75), 0, 255).astype(np.uint8)


def test_save_load_round_trip(trained):
    d, shell, X = trained
    binned = _bins(X).ravel()
    before = np.empty(ROWS, dtype=np.float32)
    shell.model.predict_chunk(binned, before)

    path = d / "model.gb"
    shell.model.save_model(str(path))
    loaded = gb_core.GradientBoostingModel(1)
    loaded.load_model(str(path))
    after = np.empty(ROWS, dtype=np.float32)
    loaded.predict_chunk(binned, after)

    np.testing.assert_array_equal(before, after)
    for getter in ("get_num_features", "get_num_trees", "get_max_depth", "get_learning_rate",
                   "get_base_score", "get_lambda", "get_gamma", "get_min_child_weight"):
        assert getattr(loaded, getter)() == getattr(shell.model, getter)(), getter

    # Saving the loaded model again reproduces the file byte for byte.
    path2 = d / "model_resaved.gb"
    loaded.save_model(str(path2))
    assert path2.read_bytes() == path.read_bytes()


def _python_prediction(model, values):
    """The shell's `predict` path: _quantize_row (main.py:56-73), then model.predict."""
    bins = main.GBEngineShell()._quantize_row([float(v) for v in values], len(values))
    return model.predict(bins)


def _cli_prediction(path, values):
    proc = run_cli(path, values)
    assert proc.returncode == 0, proc.stderr
    line = next(l for l in proc.stdout.splitlines() if l.startswith("Prediction: "))
    return line[len("Prediction: "):]


def test_python_and_cli_predict_identically(trained):
    d, shell, _ = trained
    path = d / "model_cli.gb"
    shell.model.save_model(str(path))
    rng = np.random.default_rng(11)
    rows = np.round(rng.uniform(-10.0, 10.0, size=(40, FEATURES)), 4)
    for row in rows:
        values = [f"{v:.4f}" for v in row]
        # main.cpp prints with std::cout defaults (6 significant digits, %g style).
        assert _cli_prediction(path, values) == f"{_python_prediction(shell.model, values):g}", values


def _split_model(tmp_path, split_bin):
    """One tree, one split on feature 0: bin <= split_bin -> -1, else +1."""
    nodes = [(False, 0.0, 0, split_bin, 1, 2),
             (True, -1.0, -1, 0, -1, -1),
             (True, 1.0, -1, 0, -1, -1)]
    path = tmp_path / f"split{split_bin}.gb"
    path.write_bytes(build_model_bytes(1, [nodes], learning_rate=1.0, base_score=0.0))
    model = gb_core.GradientBoostingModel(1)
    model.load_model(str(path))
    return path, model


def _training_bins(tmp_path, strings):
    """Reference bins: the training quantizer (OutOfCoreQuantizer, as used at main.py:186)."""
    path = tmp_path / "reference_values.bin"
    np.array([float(s) for s in strings], dtype=np.float32).tofile(path)
    q = gb_core.OutOfCoreQuantizer(str(path), len(strings), -10.0, 10.0)
    out = np.empty(len(strings), dtype=np.uint8)
    assert q.process_next_chunk_into(out, 0) == len(strings)
    return out


def test_python_and_cli_predict_identically_at_a_bin_edge(tmp_path):
    # Regression test for a bug this suite found. 3.333333 sits next to the 169/170 bin
    # edge: (3.333333 + 10) * 12.75 = 169.99999575 exactly, but the nearest float32 to
    # 3.333333 maps to 170.0. Training and the CLI bin in float32 and get 170; the shell
    # used to bin in float64, got 169, and predicted -1 where the CLI predicted +1.
    values = ["3.333333"]
    assert _training_bins(tmp_path, values)[0] == 170
    assert main.GBEngineShell()._quantize_row([3.333333], 1)[0] == 170
    path, model = _split_model(tmp_path, 169)
    assert _cli_prediction(path, values) == f"{_python_prediction(model, values):g}" == "1"


def test_python_bins_match_training_at_7_decimals(tmp_path):
    rng = np.random.default_rng(2024)
    uniform = rng.uniform(-10.0, 10.0, 20_000)
    # Values within a few 1e-7 of a bin edge, where float64 and float32 binning can disagree.
    edges = rng.integers(1, 256, 20_000) / 12.75 - 10.0
    near_edge = edges + rng.integers(-5, 6, 20_000) * 1e-7
    strings = [f"{v:.7f}" for v in np.concatenate([uniform, near_edge])]
    values = [float(s) for s in strings]

    reference = _training_bins(tmp_path, strings)
    python_bins = main.GBEngineShell()._quantize_row(values, len(values))
    np.testing.assert_array_equal(python_bins, reference)

    # The inputs have teeth: float64 binning (the old shell path) disagrees on some of them.
    float64_bins = np.clip((np.clip(values, -10.0, 10.0) + 10.0) * 12.75, 0, 255).astype(np.uint8)
    disputed = np.nonzero(float64_bins != reference)[0]
    assert len(disputed) > 0

    # On those inputs the CLI lands in the training bin too, and agrees with Python:
    # a split at b-1 must send bin b right (+1), a split at b must send it left (-1).
    for i in disputed[:20]:
        b, value = int(reference[i]), [strings[i]]
        for split_bin, expected in ((b - 1, "1"), (b, "-1")):
            if not 0 <= split_bin <= 255:
                continue
            path, model = _split_model(tmp_path, split_bin)
            assert _cli_prediction(path, value) == f"{_python_prediction(model, value):g}" == expected, value
