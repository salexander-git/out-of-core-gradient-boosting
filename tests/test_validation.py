"""Bad input is rejected at each boundary instead of being silently used."""
import contextlib
import io
import math

import numpy as np
import pytest

import gb_core
import main
from conftest import COUNT, HEADER, NODE, build_model_bytes, run_cli, write_synthetic


def _shell_output(shell, *commands):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for c in commands:
            shell.onecmd(c)
    return out.getvalue()


# --- Non-finite input ----------------------------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_quantizer_rejects_non_finite(tmp_path, bad):
    path = tmp_path / "f.bin"
    np.array([1.0, bad, 2.0, 3.0], dtype=np.float32).tofile(path)
    q = gb_core.OutOfCoreQuantizer(str(path), 4, -10.0, 10.0)
    with pytest.raises(RuntimeError, match="non-finite"):
        q.process_next_chunk_into(np.empty(4, dtype=np.uint8), 0)


def test_training_on_non_finite_feature_fails_and_discards_model(tmp_path):
    feat, labels, X, _ = write_synthetic(tmp_path, 200, 4, seed=1)
    X[57, 1] = np.nan
    X.tofile(feat)
    shell = main.GBEngineShell()
    out = _shell_output(shell, f"load_train {feat} {labels} 4", "train 2 3 0.1 1.0")
    assert "Training failed" in out and "non-finite" in out
    assert shell.model is None


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_cli_rejects_non_finite(tmp_path, bad):
    nodes = [(True, 0.5, -1, 0, -1, -1)]
    path = tmp_path / "m.gb"
    path.write_bytes(build_model_bytes(2, [nodes]))
    proc = run_cli(path, ["1.0", bad])
    assert proc.returncode == 1
    assert "Non-finite" in proc.stderr


def test_python_predict_rejects_non_finite():
    with pytest.raises(ValueError, match="not finite"):
        main.GBEngineShell()._quantize_row([1.0, math.nan], 2)


# --- Truncated feature files ---------------------------------------------------------

def test_quantizer_rejects_partial_float(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(np.arange(3, dtype=np.float32).tobytes()[:10])  # 2.5 floats
    q = gb_core.OutOfCoreQuantizer(str(path), 4, -10.0, 10.0)
    with pytest.raises(RuntimeError, match="partial float"):
        q.process_next_chunk_into(np.empty(4, dtype=np.uint8), 0)


def test_shell_rejects_truncated_float32_features(tmp_path):
    feat, labels, _, _ = write_synthetic(tmp_path, 100, 4, seed=2)
    feat.write_bytes(feat.read_bytes()[:-3])
    shell = main.GBEngineShell()
    out = _shell_output(shell, f"load_train {feat} {labels} 4")
    assert "[ERROR]" in out
    assert shell.features_filepath is None


def test_shell_rejects_float32_features_missing_whole_rows(tmp_path):
    feat, labels, _, _ = write_synthetic(tmp_path, 100, 4, seed=2)
    feat.write_bytes(feat.read_bytes()[:-4 * 4 * 3])  # drop 3 rows; labels still 100
    shell = main.GBEngineShell()
    out = _shell_output(shell, f"load_train {feat} {labels} 4")
    assert "Row count mismatch" in out
    assert shell.features_filepath is None


def test_shell_rejects_truncated_qbin_features(tmp_path):
    feat, labels, _, _ = write_synthetic(tmp_path, 100, 4, seed=2, qbin=True)
    feat.write_bytes(feat.read_bytes()[:-1])
    shell = main.GBEngineShell()
    out = _shell_output(shell, f"load_train_qbin {feat} {labels} 4")
    assert "does not divide evenly" in out
    assert shell.features_filepath is None


# --- Corrupted or wrong-magic model files --------------------------------------------

VALID_TREE = [(False, 0.0, 1, 100, 1, 2),
              (True, -0.5, -1, 0, -1, -1),
              (True, 0.5, -1, 0, -1, -1)]


def _load(path):
    model = gb_core.GradientBoostingModel(1)
    model.load_model(str(path))
    return model


def test_valid_hand_built_model_loads(tmp_path):
    path = tmp_path / "ok.gb"
    path.write_bytes(build_model_bytes(3, [VALID_TREE]))
    assert _load(path).get_num_trees() == 1


def test_wrong_magic_rejected(tmp_path):
    path = tmp_path / "bad.gb"
    data = bytearray(build_model_bytes(3, [VALID_TREE]))
    data[0:4] = b"GBT1"
    path.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="invalid or unsupported"):
        _load(path)
    proc = run_cli(path, ["0", "0", "0"])
    assert proc.returncode == 1 and "invalid or unsupported" in proc.stderr


def test_random_bytes_rejected(tmp_path):
    path = tmp_path / "noise.gb"
    path.write_bytes(np.random.default_rng(5).integers(0, 256, 256, dtype=np.uint8).tobytes())
    with pytest.raises(RuntimeError):
        _load(path)


def _node_offset(node_index):
    return HEADER.size + COUNT.size + node_index * NODE.size


@pytest.mark.parametrize("left, right, message", [
    (99, 2, "child index out of range|invalid left_child_idx"),
    (0, 2, "cycle"),
    (1, 1, "children must differ"),
])
def test_corrupt_tree_structure_rejected(tmp_path, left, right, message):
    path = tmp_path / "corrupt.gb"
    data = bytearray(build_model_bytes(3, [VALID_TREE]))
    root = list(NODE.unpack_from(data, _node_offset(0)))
    root[4], root[5] = left, right
    NODE.pack_into(data, _node_offset(0), *root)
    path.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match=message):
        _load(path)
    assert run_cli(path, ["0", "0", "0"]).returncode == 1


def test_split_feature_out_of_range_rejected(tmp_path):
    tree = [(False, 0.0, 7, 100, 1, 2)] + VALID_TREE[1:]  # feature 7 in a 3-feature model
    path = tmp_path / "feat.gb"
    path.write_bytes(build_model_bytes(3, [tree]))
    with pytest.raises(RuntimeError, match="invalid split_feature_idx"):
        _load(path)


def test_truncated_model_rejected(tmp_path):
    path = tmp_path / "short.gb"
    path.write_bytes(build_model_bytes(3, [VALID_TREE])[:-10])
    with pytest.raises(RuntimeError, match="failed while reading tree nodes"):
        _load(path)
