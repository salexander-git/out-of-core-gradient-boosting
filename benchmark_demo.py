import subprocess
import os
import re

THREADS = [1, 2, 4, 8]

DATASETS = [
    {
        "name": "california_qbin_mem",
        "load_cmd": "load_train_qbin_mem california_features_qbin.bin california_labels.bin 8",
        "chunk_rows": 1_000_000,
    },
    {
        "name": "yearpred_qbin_mem",
        "load_cmd": "load_train_qbin_mem yearpred_train_features_qbin.bin yearpred_train_labels.bin 90",
        "chunk_rows": 100_000,
    },
]

PATTERNS = {
    "total_time": re.compile(r"total_time=([0-9.]+)s"),
    "final_rmse": re.compile(r"final_rmse=([0-9.]+)"),
    "memory_after": re.compile(r"memory_after=([0-9.]+) MB"),
}


def run_case(dataset_cfg, thread_count):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(thread_count)

    commands = [
        dataset_cfg["load_cmd"],
        "set_verbose_levels off",
        f"set_chunk_rows {dataset_cfg['chunk_rows']}",
        "train 20 6 0.1 1.0",
    ]

    cmd = ["python", "main.py"]
    for c in commands:
        cmd.extend(["--cmd", c])

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
    )

    output = proc.stdout + "\n" + proc.stderr

    result = {
        "dataset": dataset_cfg["name"],
        "threads": thread_count,
        "returncode": proc.returncode,
        "total_time": None,
        "final_rmse": None,
        "memory_after": None,
        "raw_output": output,
    }

    for key, pattern in PATTERNS.items():
        match = pattern.search(output)
        if match:
            result[key] = float(match.group(1))

    return result


def fmt(value, decimals=2):
    if value is None:
        return "FAIL"
    return f"{value:.{decimals}f}"


def print_dataset_table(dataset_name, rows):
    print(f"\n=== {dataset_name.upper()} THREAD BENCHMARK ===")
    print("{:<8} {:<14} {:<14} {:<12}".format(
        "threads", "total_time(s)", "final_rmse", "memory(MB)"
    ))
    print("-" * 54)

    for r in rows:
        print("{:<8} {:<14} {:<14} {:<12}".format(
            r["threads"],
            fmt(r["total_time"], 2),
            fmt(r["final_rmse"], 6),
            fmt(r["memory_after"], 2),
        ))

    baseline = next((r for r in rows if r["threads"] == 1 and r["total_time"] is not None), None)
    if baseline is not None:
        print("\nSpeedup vs 1 thread")
        print("{:<8} {:<10}".format("threads", "speedup"))
        print("-" * 20)
        for r in rows:
            if r["total_time"] is None:
                speedup = "FAIL"
            else:
                speedup = f"{baseline['total_time'] / r['total_time']:.2f}"
            print("{:<8} {:<10}".format(r["threads"], speedup))


def main():
    all_results = []

    for ds in DATASETS:
        print(f"\nRunning dataset={ds['name']} ...")
        for t in THREADS:
            print(f"  threads={t} ...", end="", flush=True)
            result = run_case(ds, t)
            all_results.append(result)

            if result["returncode"] != 0:
                print(" FAIL")
            else:
                print(
                    f" done "
                    f"(time={fmt(result['total_time'])}s, "
                    f"rmse={fmt(result['final_rmse'], 6)}, "
                    f"mem={fmt(result['memory_after'])}MB)"
                )

    for ds in DATASETS:
        rows = [r for r in all_results if r["dataset"] == ds["name"]]
        print_dataset_table(ds["name"], rows)

    failures = [r for r in all_results if r["returncode"] != 0 or r["total_time"] is None]
    if failures:
        print("\n=== FAILURE DETAILS ===")
        for r in failures:
            print(f"\n--- dataset={r['dataset']} threads={r['threads']} ---")
            print(r["raw_output"])


if __name__ == "__main__":
    main()