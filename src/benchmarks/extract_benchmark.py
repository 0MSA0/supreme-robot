#!/usr/bin/env python3
"""Extract benchmark.txt into benchmark.csv."""

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "benchmark.txt"
OUTPUT = ROOT / "benchmark.csv"

GPU_RE = re.compile(r"^GPU Model:\s*(.+)$")
SECTION_RE = re.compile(r"^---\s*(.*?)\s*---$")
MODEL_RE = re.compile(r"^Processing (\S+) \((\d+)M\)")
BATCH_RE = re.compile(
    r"Batch size (\d+): "
    r"(?:Tokens/sec:\s*([\d.]+), MFU:\s*([\d.]+)|Out of memory)"
)
MAX_RE = re.compile(r"Max batch size that fit:\s*(\d+)")
BEST_RE = re.compile(r"-> best:\s*batch (\d+), ([\d.]+) tokens/s, MFU ([\d.]+)")


def extract(source: Path) -> list[dict[str, object]]:
    gpu_model = ""
    mode = "Initial"
    model = ""
    parameters = 0
    max_batch: dict[tuple[str, str], int] = {}
    best_batch: dict[tuple[str, str], int] = {}
    rows: list[dict[str, object]] = []

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        gpu_match = GPU_RE.match(line)
        if gpu_match:
            gpu_model = gpu_match.group(1)
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            mode = section_match.group(1)
            continue

        model_match = MODEL_RE.match(line)
        if model_match:
            model = model_match.group(1)
            parameters = int(model_match.group(2))
            continue

        batch_match = BATCH_RE.search(line)
        if batch_match:
            batch_size = int(batch_match.group(1))
            throughput = batch_match.group(2)
            mfu = batch_match.group(3)
            status = "success" if throughput is not None else "out_of_memory"
            rows.append(
                {
                    "gpu_model": gpu_model,
                    "mode": mode,
                    "model": model,
                    "parameters_m": parameters,
                    "batch_size": batch_size,
                    "status": status,
                    "tokens_per_second": float(throughput) if throughput else "",
                    "mfu": float(mfu) if mfu else "",
                }
            )
            continue

        max_match = MAX_RE.match(line)
        if max_match:
            max_batch[(mode, model)] = int(max_match.group(1))
            continue

        best_match = BEST_RE.match(line)
        if best_match:
            best_batch[(mode, model)] = int(best_match.group(1))

    for row in rows:
        key = (str(row["mode"]), str(row["model"]))
        batch_size = int(row["batch_size"])
        row["is_best"] = batch_size == best_batch.get(key)
        row["is_max_batch"] = batch_size == max_batch.get(key)

    return rows


def main() -> None:
    rows = extract(INPUT)
    fieldnames = [
        "gpu_model",
        "mode",
        "model",
        "parameters_m",
        "batch_size",
        "status",
        "tokens_per_second",
        "mfu",
        "is_best",
        "is_max_batch",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
