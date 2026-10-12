"""Chunk equivalence: chunk size is an I/O setting and should not change the model.

The engine has no randomness (no row or feature sampling), so "same seed" reduces to
"same data and parameters": every run below differs only in chunk size and threads.
"""
import numpy as np
import pytest

import gb_core
from conftest import final_rmse, parse_model, run_main, write_synthetic

SMALL_CHUNK_ROWS = 1_000
SINGLE_CHUNK_ROWS = 1_000_000  # main.py:37 default; larger than California's 20,640 rows
TRAIN = "train 20 6 0.1 1.0"   # the stored job: results/bench_threads_1.log:17


def _train_california(california_dir, tmp_path, chunk_rows, threads):
    model = tmp_path / f"california_chunk{chunk_rows}_t{threads}.gb"
    out = run_main([
        f"load_train {california_dir}/california_features.bin {california_dir}/california_labels.bin 8",
        f"set_chunk_rows {chunk_rows}",
        TRAIN,
        f"save {model}",
    ], cwd=tmp_path, threads=threads)
    return model.read_bytes(), final_rmse(out)


@pytest.fixture(scope="module")
def california_t1(california_dir, tmp_path_factory):
    d = tmp_path_factory.mktemp("equiv_t1")
    return (_train_california(california_dir, d, SINGLE_CHUNK_ROWS, 1),
            _train_california(california_dir, d, SMALL_CHUNK_ROWS, 1))


def _node_key(node):
    """What prediction depends on: (is_leaf, leaf value | None, split feature, split bin, children).

    An internal node keeps the weight it had as a leaf (finalize_level_splits never resets
    it, Tree.hpp:309-312) but prediction never reads it (Tree.hpp:423-429), so only leaf
    weights are compared.
    """
    is_leaf, weight, feature, split_bin, left, right = node
    return (is_leaf, weight if is_leaf else None, feature, split_bin, left, right)


def _first_difference(trees_a, trees_b):
    """Describe the first node whose split feature, split bin, children or leaf value differ."""
    for t, (a, b) in enumerate(zip(trees_a, trees_b)):
        if len(a) != len(b):
            return f"tree {t}: {len(a)} nodes vs {len(b)} nodes"
        for i, (na, nb) in enumerate(zip(a, b)):
            if _node_key(na) != _node_key(nb):
                return f"tree {t} node {i}: single-chunk {na} vs small-chunk {nb}"
    return None


# Regression tests for a bug these tests found. add_chunk used to sum each chunk into a
# fresh per-chunk float32 histogram and then add that partial sum into the level's running
# histogram. One chunk gave one long running sum, many chunks a sum of partial sums; float
# addition is not associative, so leaf values differed by a few ulps from tree 0 and one
# California split differed by tree 13 (0-indexed). The single-thread path now adds every
# row straight into the running histogram (Tree.hpp:166-191), so the sums depend on row
# order only and any chunk size gives a byte-identical model.
def test_threads1_final_rmse_matches(california_t1):
    (_, rmse_single), (_, rmse_small) = california_t1
    assert rmse_single == rmse_small


def test_threads1_every_tree_matches_exactly(california_t1):
    (single, _), (small, _) = california_t1
    _, trees_single = parse_model(single)
    _, trees_small = parse_model(small)
    assert len(trees_single) == len(trees_small) == 20
    diff = _first_difference(trees_single, trees_small)
    assert diff is None, diff


def test_threads1_saved_models_byte_identical(california_t1):
    (single, _), (small, _) = california_t1
    assert single == small


# --- Multi-threaded runs: compared within a tolerance -------------------------------
#
# Exact equality is not expected here. Each OpenMP thread sums gradients for its own
# static block of rows into private float32 histograms (Tree.hpp:193-235), and the
# per-thread totals are then added together in thread order (Tree.hpp:237-262).
# Float addition is not associative, so the bracketing of the sum (and with it the
# last bits of every histogram total) depends on how many threads ran and where the
# chunk boundaries fall. Those last-bit differences move leaf values by a few float32
# ulps and can flip a split between two near-equal candidates, so the test compares
# what the model predicts, not its bytes.
#
# Tolerances: every prediction within 1e-4 absolute, final RMSE within 1e-5.
# Measured on the synthetic set (g++ 13.3, x86-64): the largest prediction difference
# between 4 threads and the 1-thread reference was 1.9e-6, and printed RMSE differed by
# at most 1e-6. (California always runs on one thread, see below, so it now matches
# exactly.) 1.9e-6 is a few float32 ulps of predictions of magnitude 1-30, so the bounds
# leave ~50x headroom for compiler and hardware differences while staying far below any
# change a real modelling error would cause.
PRED_ATOL = 1e-4
RMSE_ATOL = 1e-5


def _predict_all(model_bytes, tmp_path, binned, num_rows):
    path = tmp_path / "predict.gb"
    path.write_bytes(model_bytes)
    model = gb_core.GradientBoostingModel(1)
    model.load_model(str(path))
    out = np.empty(num_rows, dtype=np.float32)
    model.predict_chunk(binned, out)
    return out


def _quantize_file(features_path, num_rows, num_features):
    """Bin a float32 feature file exactly as training does (OutOfCoreQuantizer, main.py:186)."""
    n = num_rows * num_features
    q = gb_core.OutOfCoreQuantizer(str(features_path), n, -10.0, 10.0)
    binned = np.empty(n, dtype=np.uint8)
    assert q.process_next_chunk_into(binned, 0) == n
    return binned


def test_threads4_california_within_tolerance(california_dir, tmp_path):
    # Note: California is 20,640 x 8 = 165,120 cells per chunk, under the 1,000,000-cell
    # threshold in Tree.hpp:42, so the engine's heuristic runs it on one thread even with
    # OMP_NUM_THREADS=4. The synthetic test below is the one that exercises 2 and 4 threads.
    single, rmse_single = _train_california(california_dir, tmp_path, SINGLE_CHUNK_ROWS, 4)
    small, rmse_small = _train_california(california_dir, tmp_path, SMALL_CHUNK_ROWS, 4)
    n = 20_640
    binned = _quantize_file(california_dir / "california_features.bin", n, 8)
    pa = _predict_all(single, tmp_path, binned, n)
    pb = _predict_all(small, tmp_path, binned, n)
    np.testing.assert_allclose(pa, pb, rtol=0, atol=PRED_ATOL)
    assert abs(float(rmse_single) - float(rmse_small)) <= RMSE_ATOL


def test_threads4_large_synthetic_within_tolerance(tmp_path):
    # 400,000 x 64 = 25.6M cells in one chunk -> 4 threads (Tree.hpp:45);
    # 50,000-row chunks = 3.2M cells -> 2 threads (Tree.hpp:43). Reference: 1 thread.
    rows, features = 400_000, 64
    feat, labels, _, _ = write_synthetic(tmp_path, rows, features, seed=7, qbin=True)

    def train(chunk_rows, threads):
        model = tmp_path / f"syn_chunk{chunk_rows}_t{threads}.gb"
        out = run_main([
            f"load_train_qbin {feat} {labels} {features}",
            f"set_chunk_rows {chunk_rows}",
            "train 5 5 0.1 1.0",
            f"save {model}",
        ], cwd=tmp_path, threads=threads)
        return model.read_bytes(), float(final_rmse(out))

    ref, rmse_ref = train(rows, 1)
    binned = np.fromfile(feat, dtype=np.uint8)
    p_ref = _predict_all(ref, tmp_path, binned, rows)
    for chunk_rows in (rows, 50_000):
        model, rmse = train(chunk_rows, 4)
        p = _predict_all(model, tmp_path, binned, rows)
        np.testing.assert_allclose(p, p_ref, rtol=0, atol=PRED_ATOL,
                                   err_msg=f"4 threads, chunk_rows={chunk_rows}")
        assert abs(rmse - rmse_ref) <= RMSE_ATOL
