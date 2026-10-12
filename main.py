import cmd
import shlex
import time
import os
import math
import numpy as np
import psutil
import gb_core
import argparse


class GBEngineShell(cmd.Cmd):
    intro = (
        "\n==================================================\n"
        "  OUT-OF-CORE GRADIENT BOOSTING ENGINE (REGRESSION)\n"
        "==================================================\n"
        "Type 'help' or '?' to list commands.\n"
    )
    prompt = "(gb-core) "

    def __init__(self):
        super().__init__()
        self.features_filepath = None
        self.labels_filepath = None
        self.num_features = None
        self.total_rows = 0

        # Modes: "float32", "qbin", "qbin_mem"
        self.features_mode = None
        self.features_qbin_mem = None

        self.labels = None
        self.predictions = None
        self.base_score = None

        self.model = None
        self.chunk_size_rows = 1_000_000
        self.verbose_levels = False

    def _require_training_data(self):
        if self.features_filepath is None or self.labels is None:
            print("[ERROR] No labeled training data loaded. Use 'load_train', 'load_train_qbin', or 'load_train_qbin_mem' first.")
            return False
        return True

    def _require_model(self):
        if self.model is None:
            print("[ERROR] No ensemble model available. Train or load a model first.")
            return False
        return True

    def _compute_rmse(self, preds, labels):
        diff = preds - labels
        return float(np.sqrt(np.mean(diff * diff)))

    def _quantize_row(self, raw_vals, num_features):
        # Bin in float32, exactly as training (OutOfCoreQuantizer.hpp:20-31) and the native
        # CLI (main.cpp:40-58) do. Float64 arithmetic here could pick the neighbouring bin
        # for a value within float32 rounding of a bin edge.
        f_min, f_max = np.float32(-10.0), np.float32(10.0)
        scale = np.float32(255.0) / (f_max - f_min)

        binned_vals = np.empty(num_features, dtype=np.uint8)
        for i, val in enumerate(raw_vals):
            with np.errstate(over="ignore"):
                val32 = np.float32(val)
            if not math.isfinite(val) or not np.isfinite(val32):
                raise ValueError(f"feature at position {i} is not finite")
            clamped = max(f_min, min(f_max, val32))
            mapped = (clamped - f_min) * scale
            mapped = max(np.float32(0.0), min(np.float32(255.0), mapped))
            binned_vals[i] = int(mapped)
        return binned_vals

    def _memory_mb(self):
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)

    def _print_train_header(self, num_trees, max_depth, learning_rate, min_child_weight):
        print("\n=== TRAINING CONFIG ===")
        print(
            f"rows={self.total_rows:,}  features={self.num_features}  rounds={num_trees}  "
            f"depth={max_depth}  lr={learning_rate}  min_child_weight={min_child_weight}"
        )
        print(
            f"chunk_rows={self.chunk_size_rows:,}  omp_threads={os.environ.get('OMP_NUM_THREADS', '(default)')}  "
            f"features_mode={self.features_mode}"
        )
        print(
            f"base_score={self.base_score:.6f}  initial_rmse={self._compute_rmse(self.predictions, self.labels):.6f}  "
            f"mem_before={self._memory_mb():.2f} MB"
        )
        print("\n{:<6} {:<12} {:<10} {:<10} {:<10}".format("round", "rmse", "time(s)", "mem(MB)", "trees"))
        print("-" * 54)

    def _print_round_row(self, round_idx, rmse, round_time, mem_mb, num_trees):
        print("{:<6} {:<12.6f} {:<10.2f} {:<10.2f} {:<10}".format(
            round_idx, rmse, round_time, mem_mb, num_trees
        ))

    def _print_final_summary(self, total_time, final_rmse):
        print("\n=== TRAINING SUMMARY ===")
        print(f"total_time={total_time:.2f}s")
        print(f"final_rmse={final_rmse:.6f}")
        print(f"num_trees={self.model.get_num_trees()}")
        print(f"memory_after={self._memory_mb():.2f} MB")

    def _load_labels_and_init_state(self, features_filepath, labels_filepath, num_features, feature_bytes_per_value):
        if not os.path.exists(features_filepath):
            print(f"[ERROR] Features file '{features_filepath}' not found.")
            return False
        if not os.path.exists(labels_filepath):
            print(f"[ERROR] Labels file '{labels_filepath}' not found.")
            return False

        feature_bytes = os.path.getsize(features_filepath)
        label_bytes = os.path.getsize(labels_filepath)

        if feature_bytes % feature_bytes_per_value != 0:
            print(f"[ERROR] Features file size is not divisible by {feature_bytes_per_value}.")
            return False
        if label_bytes % 4 != 0:
            print("[ERROR] Labels file size is not divisible by 4. Expected raw float32 binary data.")
            return False

        total_feature_elements = feature_bytes // feature_bytes_per_value
        total_labels = label_bytes // 4

        if total_feature_elements % num_features != 0:
            print("[ERROR] Features file does not divide evenly into rows for the given num_features.")
            return False

        total_rows = total_feature_elements // num_features
        if total_rows != total_labels:
            print(f"[ERROR] Row count mismatch: features imply {total_rows} rows but labels contain {total_labels} values.")
            return False

        try:
            labels = np.fromfile(labels_filepath, dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Failed to read labels file: {e}")
            return False

        self.features_filepath = features_filepath
        self.labels_filepath = labels_filepath
        self.num_features = num_features
        self.total_rows = total_rows
        self.labels = labels
        self.base_score = float(np.mean(self.labels))
        self.predictions = np.full(self.total_rows, self.base_score, dtype=np.float32)
        self.model = None

        baseline_rmse = self._compute_rmse(self.predictions, self.labels)

        print("[SUCCESS] Labeled training data loaded.")
        print(f"          Features: '{self.features_filepath}'")
        print(f"          Labels:   '{self.labels_filepath}'")
        print(f"          Rows: {self.total_rows:,} | Features: {self.num_features}")
        print(f"          Mode: {self.features_mode}")
        print(f"          Base Score (mean label): {self.base_score:.6f}")
        print(f"          Baseline RMSE: {baseline_rmse:.6f}")
        print(f"          Memory After Load: {self._memory_mb():.2f} MB")
        return True

    def _iter_binned_chunks(self):
        chunk_size_rows = self.chunk_size_rows
        chunk_elements = chunk_size_rows * self.num_features

        if self.features_mode == "qbin_mem":
            total_elements = self.total_rows * self.num_features
            data = self.features_qbin_mem
            if data is None or data.shape[0] != total_elements:
                raise RuntimeError("qbin_mem data is not loaded correctly")

            row_start = 0
            while row_start < self.total_rows:
                row_end = min(row_start + chunk_size_rows, self.total_rows)
                start_elem = row_start * self.num_features
                end_elem = row_end * self.num_features
                yield data[start_elem:end_elem], (row_end - row_start), row_start, row_end
                row_start = row_end
            return

        chunk_binned = np.empty(chunk_elements, dtype=np.uint8)

        if self.features_mode == "qbin":
            reader = gb_core.BinnedDataReader(self.features_filepath, chunk_elements)
            def next_chunk():
                return reader.read_next_chunk(chunk_binned, 0)
        elif self.features_mode == "float32":
            reader = gb_core.OutOfCoreQuantizer(self.features_filepath, chunk_elements, -10.0, 10.0)
            def next_chunk():
                return reader.process_next_chunk_into(chunk_binned, 0)
        else:
            raise RuntimeError(f"Unknown features_mode: {self.features_mode}")

        rows_processed = 0
        while True:
            elements_read = next_chunk()
            if elements_read == 0:
                break

            if elements_read % self.num_features != 0:
                raise RuntimeError("feature chunk did not divide evenly into full rows")

            rows_read = elements_read // self.num_features
            row_start = rows_processed
            row_end = rows_processed + rows_read
            yield chunk_binned[:elements_read], rows_read, row_start, row_end
            rows_processed += rows_read

    def do_set_threads(self, arg):
        """Set OpenMP thread count. Usage: set_threads <num_threads>"""
        args = shlex.split(arg)
        if len(args) != 1:
            print("[ERROR] Usage: set_threads <num_threads>")
            return

        try:
            n = int(args[0])
            if n <= 0:
                raise ValueError
        except ValueError:
            print("[ERROR] num_threads must be a positive integer.")
            return

        os.environ["OMP_NUM_THREADS"] = str(n)
        print(f"[SUCCESS] Set OMP_NUM_THREADS={n}")

    def do_set_chunk_rows(self, arg):
        """Set chunk size in rows. Usage: set_chunk_rows <rows>"""
        args = shlex.split(arg)
        if len(args) != 1:
            print("[ERROR] Usage: set_chunk_rows <rows>")
            return

        try:
            n = int(args[0])
            if n <= 0:
                raise ValueError
        except ValueError:
            print("[ERROR] rows must be a positive integer.")
            return

        self.chunk_size_rows = n
        print(f"[SUCCESS] chunk_size_rows set to {n}")

    def do_set_verbose_levels(self, arg):
        """Enable or disable per-level output. Usage: set_verbose_levels <on|off>"""
        args = shlex.split(arg)
        if len(args) != 1 or args[0].lower() not in {"on", "off"}:
            print("[ERROR] Usage: set_verbose_levels <on|off>")
            return

        self.verbose_levels = (args[0].lower() == "on")
        print(f"[SUCCESS] verbose_levels set to {self.verbose_levels}")

    def do_load_train(self, arg):
        """
        Load float32 feature data and float32 labels.
        Usage: load_train <features.bin> <labels.bin> <num_features>
        """
        args = shlex.split(arg)
        if len(args) != 3:
            print("[ERROR] Usage: load_train <features.bin> <labels.bin> <num_features>")
            return

        features_filepath = args[0]
        labels_filepath = args[1]

        try:
            num_features = int(args[2])
            if num_features <= 0:
                raise ValueError
        except ValueError:
            print("[ERROR] num_features must be a positive integer.")
            return

        self.features_mode = "float32"
        self.features_qbin_mem = None
        self._load_labels_and_init_state(features_filepath, labels_filepath, num_features, feature_bytes_per_value=4)

    def do_load_train_qbin(self, arg):
        """
        Load uint8 pre-quantized feature data and float32 labels.
        Usage: load_train_qbin <features_qbin.bin> <labels.bin> <num_features>
        """
        args = shlex.split(arg)
        if len(args) != 3:
            print("[ERROR] Usage: load_train_qbin <features_qbin.bin> <labels.bin> <num_features>")
            return

        features_filepath = args[0]
        labels_filepath = args[1]

        try:
            num_features = int(args[2])
            if num_features <= 0:
                raise ValueError
        except ValueError:
            print("[ERROR] num_features must be a positive integer.")
            return

        self.features_mode = "qbin"
        self.features_qbin_mem = None
        self._load_labels_and_init_state(features_filepath, labels_filepath, num_features, feature_bytes_per_value=1)

    def do_load_train_qbin_mem(self, arg):
        """
        Load uint8 pre-quantized feature data into RAM and float32 labels.
        Usage: load_train_qbin_mem <features_qbin.bin> <labels.bin> <num_features>
        """
        args = shlex.split(arg)
        if len(args) != 3:
            print("[ERROR] Usage: load_train_qbin_mem <features_qbin.bin> <labels.bin> <num_features>")
            return

        features_filepath = args[0]
        labels_filepath = args[1]

        try:
            num_features = int(args[2])
            if num_features <= 0:
                raise ValueError
        except ValueError:
            print("[ERROR] num_features must be a positive integer.")
            return

        self.features_mode = "qbin_mem"
        ok = self._load_labels_and_init_state(features_filepath, labels_filepath, num_features, feature_bytes_per_value=1)
        if not ok:
            self.features_qbin_mem = None
            return

        try:
            self.features_qbin_mem = np.fromfile(features_filepath, dtype=np.uint8)
        except Exception as e:
            self.features_qbin_mem = None
            print(f"[ERROR] Failed to load qbin features into memory: {e}")
            return

        expected = self.total_rows * self.num_features
        actual = 0 if self.features_qbin_mem is None else self.features_qbin_mem.shape[0]
        if actual != expected:
            self.features_qbin_mem = None
            print(f"[ERROR] qbin memory load mismatch: expected {expected}, got {actual}")
            return

        print(f"          qbin cached in RAM: {self.features_qbin_mem.nbytes / (1024 * 1024):.2f} MB")
        print(f"          Memory After qbin RAM Cache: {self._memory_mb():.2f} MB")

    def _apply_tree_predictions(self, tree):
        lr = self.model.get_learning_rate()

        for chunk_binned, rows_read, row_start, row_end in self._iter_binned_chunks():
            chunk_outputs = np.empty(rows_read, dtype=np.float32)
            tree.predict_chunk(chunk_binned, chunk_outputs)
            self.predictions[row_start:row_end] += lr * chunk_outputs

    def do_train(self, arg):
        """
        Train a gradient boosting ensemble for squared-error regression.
        Usage: train <num_trees> [max_depth] [learning_rate] [min_child_weight]
        Example: train 20 6 0.1 1.0
        """
        if not self._require_training_data():
            return

        args = shlex.split(arg)
        try:
            num_trees = int(args[0]) if len(args) > 0 else 10
            max_depth = int(args[1]) if len(args) > 1 else 6
            learning_rate = float(args[2]) if len(args) > 2 else 0.1
            min_child_weight = float(args[3]) if len(args) > 3 else 1.0

            if (num_trees <= 0 or max_depth <= 0 or learning_rate <= 0.0 or
                    min_child_weight < 0.0 or
                    not math.isfinite(learning_rate) or
                    not math.isfinite(min_child_weight)):
                raise ValueError
        except ValueError:
            print("[ERROR] Usage: train <num_trees> [max_depth] [learning_rate] [min_child_weight]")
            print("        num_trees > 0, max_depth > 0, learning_rate > 0, min_child_weight >= 0")
            return

        self.predictions = np.full(self.total_rows, self.base_score, dtype=np.float32)
        self.model = gb_core.GradientBoostingModel(
            self.num_features,
            max_depth,
            min_child_weight,
            1.0,
            0.0,
            learning_rate,
            self.base_score
        )

        chunk_hess = np.ones(self.chunk_size_rows, dtype=np.float32)
        self._print_train_header(num_trees, max_depth, learning_rate, min_child_weight)

        total_start = time.time()

        try:
            for round_idx in range(num_trees):
                round_start = time.time()

                tree = gb_core.GradientTree(
                    self.num_features,
                    max_depth,
                    min_child_weight
                )
                tree.init_tree()

                for level in range(max_depth):
                    if self.verbose_levels:
                        print(f"  level {level + 1}/{max_depth}")

                    tree.start_new_level()

                    for chunk_binned, rows_read, row_start, row_end in self._iter_binned_chunks():
                        gradients = self.predictions[row_start:row_end] - self.labels[row_start:row_end]
                        tree.add_chunk(
                            chunk_binned,
                            gradients,
                            chunk_hess[:rows_read]
                        )

                    tree.finalize_level_splits()

                self.model.add_tree(tree)
                self._apply_tree_predictions(tree)

                round_rmse = self._compute_rmse(self.predictions, self.labels)
                round_time = time.time() - round_start
                mem_mb = self._memory_mb()

                self._print_round_row(
                    round_idx + 1,
                    round_rmse,
                    round_time,
                    mem_mb,
                    self.model.get_num_trees()
                )

            total_time = time.time() - total_start
            final_rmse = self._compute_rmse(self.predictions, self.labels)
            self._print_final_summary(total_time, final_rmse)

        except Exception as e:
            self.model = None
            print(f"[ERROR] Training failed: {e}")

    def do_eval_train(self, arg):
        """Report training RMSE for the current predictions."""
        if not self._require_training_data():
            return
        rmse = self._compute_rmse(self.predictions, self.labels)
        print(f"[TRAIN EVAL] RMSE: {rmse:.6f}")
        print(f"[TRAIN EVAL] Memory: {self._memory_mb():.2f} MB")

    def do_save(self, arg):
        """Save the current ensemble model. Usage: save <filepath>"""
        if not self._require_model():
            return

        args = shlex.split(arg)
        if len(args) != 1:
            print("[ERROR] Usage: save <filepath>")
            return

        try:
            self.model.save_model(args[0])
            print(f"[SUCCESS] Ensemble model serialized to '{args[0]}'.")
            print(f"          Trees: {self.model.get_num_trees()}")
        except Exception as e:
            print(f"[ERROR] Save failed: {e}")

    def do_load_model(self, arg):
        """
        Load a previously trained ensemble model.
        Usage: load_model <filepath>
        """
        args = shlex.split(arg)
        if len(args) != 1:
            print("[ERROR] Usage: load_model <filepath>")
            return

        filepath = args[0]
        if not os.path.exists(filepath):
            print(f"[ERROR] File '{filepath}' not found.")
            return

        try:
            self.model = gb_core.GradientBoostingModel(1)
            self.model.load_model(filepath)

            self.num_features = self.model.get_num_features()
            self.base_score = self.model.get_base_score()

            print(f"[SUCCESS] Ensemble model loaded '{filepath}'.")
            print(f"          Features: {self.num_features}")
            print(f"          Trees: {self.model.get_num_trees()}")
            print(f"          Learning Rate: {self.model.get_learning_rate():.6f}")
            print(f"          Base Score: {self.model.get_base_score():.6f}")
            print(f"          Memory After Load Model: {self._memory_mb():.2f} MB")
        except Exception as e:
            self.model = None
            print(f"[ERROR] Load failed: {e}")

    def do_predict(self, arg):
        """
        Predict using the ensemble model.
        Usage: predict <val1> [val2] ... [valN]
        """
        if not self._require_model():
            return

        model_features = self.model.get_num_features()
        args = shlex.split(arg)

        if len(args) != model_features:
            print(f"[ERROR] Model expects {model_features} features, but you provided {len(args)}.")
            return

        try:
            raw_vals = [float(x) for x in args]
            binned_vals = self._quantize_row(raw_vals, model_features)
        except ValueError as e:
            print(f"[ERROR] {e}")
            return

        try:
            start_time = time.time()
            raw_sum = self.model.predict_raw_sum(binned_vals)
            prediction = self.model.predict(binned_vals)
            exec_time = (time.time() - start_time) * 1000.0

            print(f"[INFERENCE RESULT] Input: {raw_vals}")
            print(f"                   Quantized Bins: {binned_vals.tolist()}")
            print(f"                   Base Score: {self.model.get_base_score():.6f}")
            print(f"                   Raw Tree Sum: {raw_sum:.6f}")
            print(f"                   Learning Rate: {self.model.get_learning_rate():.6f}")
            print(f"                   Prediction: {prediction:.6f}")
            print(f"                   Execution Time: {exec_time:.4f} ms")
            print(f"                   Memory: {self._memory_mb():.2f} MB")
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")

    def do_status(self, arg):
        """Show current shell state."""
        print("=== SHELL STATUS ===")
        print(f"verbose_lvls : {self.verbose_levels}")
        print(f"features file : {self.features_filepath}")
        print(f"labels file   : {self.labels_filepath}")
        print(f"num_features  : {self.num_features}")
        print(f"total_rows    : {self.total_rows}")
        print(f"features_mode : {self.features_mode}")
        print(f"labels_loaded : {self.labels is not None}")
        print(f"base_score    : {self.base_score}")
        print(f"model_loaded  : {self.model is not None}")
        print(f"chunk_rows    : {self.chunk_size_rows}")
        print(f"omp_threads   : {os.environ.get('OMP_NUM_THREADS', '(default)')}")
        print(f"memory_mb     : {self._memory_mb():.2f}")

        if self.features_qbin_mem is not None:
            print(f"qbin_mem_MB   : {self.features_qbin_mem.nbytes / (1024 * 1024):.2f}")

        if self.labels is not None and self.predictions is not None:
            try:
                print(f"train_rmse    : {self._compute_rmse(self.predictions, self.labels):.6f}")
            except Exception as e:
                print(f"train_rmse    : error ({e})")

        if self.model is not None:
            try:
                print(f"model_features: {self.model.get_num_features()}")
                print(f"num_trees     : {self.model.get_num_trees()}")
                print(f"learning_rate : {self.model.get_learning_rate():.6f}")
            except Exception as e:
                print(f"model_error   : {e}")

    def do_exit(self, arg):
        """Exit the shell."""
        print("Shutting down engine.")
        return True

    def do_EOF(self, arg):
        print()
        return self.do_exit(arg)


def run_batch_commands(shell, commands):
    for command in commands:
        if not command.strip():
            continue
        print(f"(gb-core) {command}")
        stop = shell.onecmd(command)
        if stop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Out-of-Core Gradient Boosting Engine (Regression)"
    )
    parser.add_argument(
        "--cmd",
        action="append",
        default=[],
        help="Command to execute non-interactively. Can be provided multiple times."
    )

    args = parser.parse_args()

    shell = GBEngineShell()

    if args.cmd:
        run_batch_commands(shell, args.cmd)
    else:
        shell.cmdloop()