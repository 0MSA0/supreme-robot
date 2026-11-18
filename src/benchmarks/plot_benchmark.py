#!/usr/bin/env python3
"""Plot benchmark throughput and MFU from benchmark.csv."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_data(csv_path: Path) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["status"] == "success":
                row["batch_size"] = int(row["batch_size"])
                row["tokens_per_second"] = float(row["tokens_per_second"])
                row["mfu"] = float(row["mfu"])
                groups[f'{row["mode"]} / {row["model"]}'].append(row)
    return groups


def plot(csv_path: Path, output_path: Path) -> None:
    groups = load_data(csv_path)
    if not groups:
        raise ValueError(f"No successful benchmark rows found in {csv_path}")

    mode_colors = {
        "Current impl": "tab:blue",
        "BFloat16": "tab:orange",
        "Torch Modules": "tab:green",
        "Compiled torch": "tab:red",
    }
    model_styles = {
        "gpt-small": ("-", "o"),
        "gpt-medium": ("--", "s"),
        "gpt-large": ("-.", "^"),
        "gpt-xl": (":", "D"),
    }

    fig, axis = plt.subplots(figsize=(12, 7))
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda item: item["batch_size"])
        batch_sizes = [row["batch_size"] for row in rows]
        throughput = [row["tokens_per_second"] for row in rows]
        mode, model = key.split(" / ", 1)
        line_style, marker = model_styles.get(model, ("-", "o"))
        axis.plot(
            batch_sizes,
            throughput,
            color=mode_colors.get(mode, "tab:gray"),
            linestyle=line_style,
            marker=marker,
            label=key,
        )

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        successful_batch_sizes = {
            int(row["batch_size"])
            for row in csv.DictReader(csv_file)
            if row["status"] == "success"
        }
    power_of_two_ticks = [
        2**exponent
        for exponent in range(12)
        if 2**exponent <= max(successful_batch_sizes)
    ]
    ticks = sorted(successful_batch_sizes | set(power_of_two_ticks))
    axis.set_xticks(ticks)
    axis.set_xlim(0, max(successful_batch_sizes) * 1.05)

    axis.set_title("Benchmark throughput")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Tokens/sec")
    axis.grid(alpha=0.25, which="both")
    axis.legend(ncols=2, fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=Path, default=Path(__file__).with_name("benchmark.csv"))
    parser.add_argument("-o", "--output", type=Path, default=Path(__file__).with_name("benchmark_plot.png"))
    args = parser.parse_args()
    plot(args.csv, args.output)


if __name__ == "__main__":
    main()
