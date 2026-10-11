# Out-of-Core Gradient Boosting Engine

[![CI](https://github.com/salexander-git/out-of-core-gradient-boosting/actions/workflows/ci.yml/badge.svg)](https://github.com/salexander-git/out-of-core-gradient-boosting/actions/workflows/ci.yml)

A gradient boosting engine for squared-error regression, written from scratch:

- **C++11 core** (header-only) for quantization, histogram building, split search, tree growth, prediction and model I/O.
- **OpenMP** parallel histogram accumulation ([`Tree.hpp:197-231`](Tree.hpp#L197-L231)).
- **pybind11** bindings exposing the core as the Python module `gb_core` ([`engine_wrapper.cpp:12`](engine_wrapper.cpp#L12)).
- **Python boosting loop** and interactive shell that stream feature chunks from disk into the C++ core ([`main.py:355-445`](main.py#L355-L445)).
- **Native inference CLI** that loads a saved model and predicts without Python ([`main.cpp`](main.cpp)).

Features are quantized to one byte each (256 codes over a fixed range of [-10, 10]). Each tree grows level by level from per-bin gradient/hessian histograms, and splits are chosen with second-order (Newton) gain.

## Repository layout

| File | Role |
|---|---|
| [`OutOfCoreQuantizer.hpp`](OutOfCoreQuantizer.hpp) | Streams a float32 file in fixed-size chunks and maps each value to a `uint8` code |
| [`BinnedDataReader.hpp`](BinnedDataReader.hpp) | Streams an already-quantized `uint8` file in fixed-size chunks |
| [`HistogramBuilder.hpp`](HistogramBuilder.hpp) | 256-bin histogram structs and per-thread histogram state |
| [`SplitEvaluator.hpp`](SplitEvaluator.hpp) | Regularized second-order split gain and leaf weights |
| [`Tree.hpp`](Tree.hpp) | Level-wise tree growth, OpenMP accumulation, structural validation, single-tree format (GBT1) |
| [`GradientBoostingModel.hpp`](GradientBoostingModel.hpp) | Ensemble container, prediction, ensemble format (GBE1) |
| [`engine_wrapper.cpp`](engine_wrapper.cpp) | pybind11 bindings (`gb_core`) |
| [`main.py`](main.py) | Boosting loop and `cmd.Cmd` shell, also scriptable with `--cmd` |
| [`main.cpp`](main.cpp) | Native inference CLI (`gb_engine`) |
| [`benchmark_demo.py`](benchmark_demo.py) | Thread-count benchmark harness |
| [`prepare_california_data.py`](prepare_california_data.py), [`prepare_yearpredictionmsd.py`](prepare_yearpredictionmsd.py) | Dataset preparation |
| [`CMakeLists.txt`](CMakeLists.txt) | Builds `gb_core` and `gb_engine` |
| [`tests/`](tests) | pytest suite; fixtures are generated at test time |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | CI: build with CMake and run pytest |
| [`results/`](results) | The stored training logs behind the results below |

## Design decisions

### 1. Chunked streaming
`OutOfCoreQuantizer` reads a flat float32 file sequentially. It allocates one float buffer and one byte buffer, each `chunk_size` long, once in the constructor ([`OutOfCoreQuantizer.hpp:50-51`](OutOfCoreQuantizer.hpp#L50-L51)). Each call reads the next chunk ([`:106-107`](OutOfCoreQuantizer.hpp#L106-L107)) and writes codes straight into a caller-owned NumPy buffer ([`:121-123`](OutOfCoreQuantizer.hpp#L121-L123)). `BinnedDataReader` does the same for pre-quantized files ([`BinnedDataReader.hpp:30-46`](BinnedDataReader.hpp#L30-L46)). On the Python side, one generator yields chunks for three feature modes ([`main.py:160-205`](main.py#L160-L205)):

- `float32`: quantize on the fly.
- `qbin`: stream pre-quantized bytes.
- `qbin_mem`: load the whole quantized file into RAM. This mode is explicitly not out-of-core ([`main.py:164-177`](main.py#L164-L177), [`:330-331`](main.py#L330-L331)).

### 2. Level-wise growth, rows re-routed instead of per-node row lists
The tree stores no row membership. On every `add_chunk`, each row walks from the root to its current leaf by comparing its bin with each node's `split_bin` ([`Tree.hpp:176-185`](Tree.hpp#L176-L185)). Between chunks, the only training state the tree holds is one histogram per (node, feature) ([`Tree.hpp:34`](Tree.hpp#L34), [`:147`](Tree.hpp#L147)), so memory does not grow with the number of rows. The cost is one pass over the features per level plus one prediction pass per tree ([`main.py:408-425`](main.py#L408-L425)).

### 3. Thread-private histograms, merged afterwards
Each OpenMP thread gets its own `ThreadHistogramState`, laid out as node → feature → 256 bins ([`HistogramBuilder.hpp:57-73`](HistogramBuilder.hpp#L57-L73), [`Tree.hpp:166-170`](Tree.hpp#L166-L170)). Rows are divided with `#pragma omp parallel` / `#pragma omp for schedule(static)` ([`Tree.hpp:198`](Tree.hpp#L198), [`:208`](Tree.hpp#L208)), and each thread writes only to its own state ([`:205`](Tree.hpp#L205)). A separate merge sums the per-thread states. It runs in parallel over (node, feature) pairs when there are at least 64 pairs ([`Tree.hpp:234-259`](Tree.hpp#L234-L259)). No atomics or locks are used. A work-size heuristic caps the thread count at 1, 2, 3 or 4 depending on rows × features ([`Tree.hpp:36-50`](Tree.hpp#L36-L50)).

### 4. Caller-driven level protocol
`GradientTree` has no `fit()`. The caller drives each level through `init_tree` → (`start_new_level` → `add_chunk` × N → `finalize_level_splits`) × depth ([`Tree.hpp:130-338`](Tree.hpp#L130-L338)). This lets Python own the data source and the boosting state while C++ does the per-chunk work. The loop in [`main.py:398-425`](main.py#L398-L425) handles the boosting:

- It initializes predictions to the mean label ([`main.py:144`](main.py#L144)).
- It computes squared-error gradients `prediction − label` per chunk, with hessian 1 ([`main.py:392`](main.py#L392), [`:415`](main.py#L415)).
- It applies each finished tree through the C++ `predict_chunk` ([`main.py:347-353`](main.py#L347-L353)).

Splits come from one prefix-sum scan per histogram ([`SplitEvaluator.hpp:55-76`](SplitEvaluator.hpp#L55-L76)). The leaf weight is −G/(H+λ) and the gain is ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ ([`SplitEvaluator.hpp:20-26`](SplitEvaluator.hpp#L20-L26), [`:67`](SplitEvaluator.hpp#L67)).

### 5. Tagged binary model format
The ensemble file (GBE1) has three parts ([`GradientBoostingModel.hpp:148-170`](GradientBoostingModel.hpp#L148-L170)):

1. A 32-bit magic number, `0x47424531`.
2. The model metadata: feature count, depth, `min_child_weight`, λ, γ, learning rate and base score.
3. A tree count, followed by each tree as a node count and a flat node array.

The loader handles a bad file in three steps:

- It rejects a wrong magic number before reading anything else ([`:183-187`](GradientBoostingModel.hpp#L183-L187)).
- It validates the metadata ([`:197`](GradientBoostingModel.hpp#L197)).
- It validates each tree's structure ([`Tree.hpp:342-349`](Tree.hpp#L342-L349)).

Prediction is `base_score + learning_rate · Σ tree(x)` ([`GradientBoostingModel.hpp:121-124`](GradientBoostingModel.hpp#L121-L124)). Single trees have their own tag, GBT1 `0x47425431` ([`Tree.hpp:362`](Tree.hpp#L362)).

### 6. Validation at every boundary
- **Input values:** the quantizer rejects non-finite values, clamps twice (input range, then code range) and uses a NaN-safe range check ([`OutOfCoreQuantizer.hpp:21-29`](OutOfCoreQuantizer.hpp#L21-L29), [`:40-42`](OutOfCoreQuantizer.hpp#L40-L42)). The CLI and the shell apply the same non-finite rule ([`main.cpp:46-49`](main.cpp#L46-L49), [`main.py:62-63`](main.py#L62-L63)).
- **Stream reads:** reads reject a file that ends in a partial float ([`OutOfCoreQuantizer.hpp:114-116`](OutOfCoreQuantizer.hpp#L114-L116)).
- **Python ↔ C++:** bindings check the dimensions, writability, capacity and length of every NumPy buffer before passing a raw pointer ([`engine_wrapper.cpp:20-41`](engine_wrapper.cpp#L20-L41), [`:95-111`](engine_wrapper.cpp#L95-L111)).
- **Tree structure:** a DFS from the root rejects out-of-range child indices, cycles, leaves with children, invalid feature indices and identical children ([`Tree.hpp:52-105`](Tree.hpp#L52-L105)). It runs before tree operations, on load and on save.

## Result

The stored result is California Housing, 20,640 rows × 8 features, 20 trees, depth 6, learning rate 0.1 ([`results/bench_threads_1.log:10`](results/bench_threads_1.log#L10), [`:17`](results/bench_threads_1.log#L17)).

| | Training RMSE |
|---|---|
| Baseline (predict the mean) | 1.153928 ([`:12`](results/bench_threads_1.log#L12)) |
| After 20 trees | 0.598397 ([`:46`](results/bench_threads_1.log#L46)) |

Total training time was 0.99 s ([`:45`](results/bench_threads_1.log#L45)). The per-round RMSE is at [`:21-42`](results/bench_threads_1.log#L21-L42). This is a **training-set** fit; see Limitations.

**Multi-chunk run.** The same job was rerun with 1,000-row chunks, so each pass streams the data in 21 chunks ([`results/california_chunk1000_threads1.log:1-4`](results/california_chunk1000_threads1.log#L1-L4), [`:22`](results/california_chunk1000_threads1.log#L22)). The log header records the commit, chunk size, compiler and exact command. Every per-round RMSE matches the single-chunk log to six decimals ([`:27-46`](results/california_chunk1000_threads1.log#L27-L46)), and the final training RMSE is again 0.598397 ([`:50`](results/california_chunk1000_threads1.log#L50)). The saved models are close but not bit-identical; see Limitations.

## Build

You need Linux or WSL, g++ with OpenMP, and Python 3 with the packages the code imports ([`main.py:6-8`](main.py#L6-L8), [`prepare_california_data.py:1-3`](prepare_california_data.py#L1-L3)).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install cmake numpy psutil pybind11 scikit-learn pytest

cmake -S . -B build -DPython_EXECUTABLE="$(which python)"
cmake --build build -j
```

[`CMakeLists.txt`](CMakeLists.txt) builds the extension `gb_core` and the CLI `gb_engine` into `build/`, both as C++11 with `-Wall` and OpenMP ([`:4`](CMakeLists.txt#L4), [`:21-33`](CMakeLists.txt#L21-L33)). It finds pip's pybind11 through the Python interpreter ([`:13-20`](CMakeLists.txt#L13-L20)). For the commands under Run, put the build directory on `PYTHONPATH` (`export PYTHONPATH=build`) and call `build/gb_engine`.

**Alternative: manual `g++`.** These commands build the same two targets in the repository root:

```bash
# Python extension: gb_core
g++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(python -m pybind11 --includes) \
    engine_wrapper.cpp -o gb_core$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Native inference CLI: gb_engine
g++ -O3 -Wall -std=c++11 -fopenmp main.cpp -o gb_engine
```

## Test

```bash
python -m pytest -v        # after the CMake build; set GB_BUILD_DIR to use another build directory
```

The tests import `gb_core` from `build/` and run `build/gb_engine` ([`tests/conftest.py:19-22`](tests/conftest.py#L19-L22)). Every fixture is generated at test time ([`tests/conftest.py:84-99`](tests/conftest.py#L84-L99)). California Housing is downloaded by scikit-learn through the repo's own prep script ([`tests/conftest.py:102-116`](tests/conftest.py#L102-L116)) and is never committed. CI runs the same build and suite on Ubuntu 24.04 with gcc and Python 3.12 ([`.github/workflows/ci.yml:9-39`](.github/workflows/ci.yml#L9-L39)).

| Area | Tests |
|---|---|
| **Chunk equivalence** | California with 1 thread, single chunk vs 1,000-row chunks: final RMSE, every tree's splits and leaf values, and the saved bytes ([`test_chunk_equivalence.py:65-81`](tests/test_chunk_equivalence.py#L65-L81)). With 4 threads, predictions are compared within a stated tolerance; the comment explains why exact equality isn't expected ([`:86-102`](tests/test_chunk_equivalence.py#L86-L102), [`:124-162`](tests/test_chunk_equivalence.py#L124-L162)). |
| **Model I/O** | Save/load round trip with identical predictions and byte-identical re-save ([`test_model_io.py:32-53`](tests/test_model_io.py#L32-L53)) |
| **Python vs CLI** | Identical predictions from the shell's path and `gb_engine` on the same model ([`test_model_io.py:69-99`](tests/test_model_io.py#L69-L99)) |
| **Rejection of bad input** | Non-finite values, truncated feature files, and wrong-magic, random, truncated or structurally corrupt model files ([`test_validation.py`](tests/test_validation.py)) |

**Known failing tests.** Three tests fail against the current engine and are deliberately left strict:

- **Chunk equivalence (two tests):** the exact 1-thread tests fail because chunking changes the float32 summation order ([`test_chunk_equivalence.py:57-64`](tests/test_chunk_equivalence.py#L57-L64)).
- **Bin edge (one test):** the Python and CLI paths disagree on one input ([`test_model_io.py:86-89`](tests/test_model_io.py#L86-L89)).

Both are listed under Limitations.

## Run

```bash
# 1. Prepare California Housing: features min-max scaled to [-10, 10]
python prepare_california_data.py

# 2. Train 20 trees of depth 6 and save the ensemble
python main.py --cmd "load_train california_features.bin california_labels.bin 8" \
               --cmd "train 20 6 0.1 1.0" \
               --cmd "save california.gb"

# 3. Predict one row with the native CLI (8 values, already scaled to [-10, 10])
./gb_engine california.gb 0 0 0 0 0 0 0 0
```

- **Data preparation:** step 1 writes `california_features.bin` (float32), `california_features_qbin.bin` (uint8) and `california_labels.bin` ([`prepare_california_data.py:14-20`](prepare_california_data.py#L14-L20)).
- **CLI output:** `gb_engine` prints `Prediction:` and `ExecTimeMs:` ([`main.cpp:67-68`](main.cpp#L67-L68)).
- **Interactive shell:** run `python main.py` with no arguments. Commands include `load_train`, `load_train_qbin`, `load_train_qbin_mem`, `set_chunk_rows`, `train`, `eval_train`, `save`, `load_model`, `predict` and `status`.
- **Threads:** set `OMP_NUM_THREADS` before starting Python, as [`benchmark_demo.py:28-29`](benchmark_demo.py#L28-L29) does. The shell's `set_threads` only changes the environment variable after `gb_core` has been imported ([`main.py:8`](main.py#L8), [`:222`](main.py#L222)).
- **Thread benchmark:** `python benchmark_demo.py` runs 1/2/4/8 threads on California and YearPredictionMSD ([`benchmark_demo.py:5-18`](benchmark_demo.py#L5-L18)). It needs `prepare_yearpredictionmsd.py`, which downloads the dataset from the UCI repository ([`prepare_yearpredictionmsd.py:10`](prepare_yearpredictionmsd.py#L10)).

## Limitations

- **There is no held-out evaluation.** Every reported RMSE is training RMSE: it is computed against the training labels ([`main.py:427`](main.py#L427), [`:440`](main.py#L440)). `prepare_yearpredictionmsd.py` writes test-split files ([`:123-125`](prepare_yearpredictionmsd.py#L123-L125)), but no code reads them.
- **Thread scaling measured flat.** Total training time on the California run was **0.99 / 0.98 / 0.99 / 1.04 s at 1 / 2 / 4 / 8 threads** ([`results/bench_threads_{1,2,4,8}.log:45`](results)), with an identical final RMSE in all four runs ([`:46`](results/bench_threads_1.log#L46)). Two things limit what these logs can show:
  - Under the work-size heuristic in the final code, this workload (20,640 × 8 = 165,120 cells) runs on one thread whatever the setting ([`Tree.hpp:42`](Tree.hpp#L42)).
  - The logs come from an earlier `main.py`: their config line ([`results/bench_threads_1.log:18`](results/bench_threads_1.log#L18)) lacks the `features_mode` field the final code prints ([`main.py:81-82`](main.py#L81-L82)). Which thread heuristic they ran under is not recorded.
- **Chunk size changes the model in its last bits.** Each chunk is summed into a fresh float32 histogram, and that partial sum is then added into the level's running total ([`Tree.hpp:166-170`](Tree.hpp#L166-L170), [`:240-242`](Tree.hpp#L240-L242)). A different chunk size therefore brackets the float sums differently:
  - On California with 1,000-row chunks, leaf values differ by a few ulps from the first tree, and one split choice differs ([`tests/test_chunk_equivalence.py:57-64`](tests/test_chunk_equivalence.py#L57-L64)).
  - Training RMSE is unaffected at six decimals ([`results/california_chunk1000_threads1.log:27-50`](results/california_chunk1000_threads1.log#L27-L50)).
  - The exact-equivalence tests fail ([`:70-81`](tests/test_chunk_equivalence.py#L70-L81)).
- **Python and CLI predictions can disagree at a bin edge.** The shell parses and bins inputs in float64 ([`main.py:520`](main.py#L520), [`:56-68`](main.py#L56-L68)), and the CLI does both in float32 ([`main.cpp:40-58`](main.cpp#L40-L58)). An input within float32 rounding of a bin boundary can land in different bins: `3.333333` goes to bin 169 in Python and bin 170 in the CLI. When a tree splits at that boundary, the two paths predict differently ([`tests/test_model_io.py:81-99`](tests/test_model_io.py#L81-L99)).
- **Validation sits in the prediction hot path.** `predict_chunk` validates the tree once and then calls `predict`, which validates it again for every row ([`Tree.hpp:420`](Tree.hpp#L420), [`:440-444`](Tree.hpp#L440-L444)).
- **The model format is not portable.** It writes native-endian integers, `size_t` counts and raw `TreeNode` struct bytes ([`GradientBoostingModel.hpp:149`](GradientBoostingModel.hpp#L149), [`:160`](GradientBoostingModel.hpp#L160), [`:169`](GradientBoostingModel.hpp#L169)).
- **Input handling is minimal.**
  - Inputs must already be scaled to [-10, 10], and the scaler is not saved with the model ([`main.cpp:40-42`](main.cpp#L40-L42), [`main.py:57`](main.py#L57)).
  - There is no missing-value handling: a non-finite value aborts training ([`OutOfCoreQuantizer.hpp:21-23`](OutOfCoreQuantizer.hpp#L21-L23), [`main.py:443-445`](main.py#L443-L445)).

## License

[MIT](LICENSE)
