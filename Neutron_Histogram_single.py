#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_hist_csv(csv_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    cols = set(data.dtype.names or [])
    if {"x", "content"}.issubset(cols):
        return data["x"], data["content"]
    if {"bin_center", "count"}.issubset(cols):
        return data["bin_center"], data["count"]
    raise RuntimeError(
        f"Unexpected CSV headers in {csv_path}. Expected x/content or bin_center/count."
    )


def main():
    parser = argparse.ArgumentParser(
        description="1D histogram plot: deposited energy (x) vs counts (y) from one CSV."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to a single histogram CSV file.",
    )
    parser.add_argument(
        "--out",
        default="/home/claire/allpix-squared/Neutrons/ROOT_plots/deposited_energy_hist_single.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=0.0,
        help="Minimum deposited energy (keV) to include.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=6000.0,
        help="Maximum deposited energy (keV) to include.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=100.0,
        help="Deposited energy bin width in keV.",
    )
    parser.add_argument(
        "--title",
        default="Deposited Energy vs Counts",
        help="Plot title.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    x, y = read_hist_csv(csv_path)

    xmin = args.xmin
    xmax = args.xmax
    bin_width = args.bin_width
    if xmax <= xmin:
        raise RuntimeError("--xmax must be greater than --xmin")
    if bin_width <= 0:
        raise RuntimeError("--bin-width must be > 0")

    edges = np.arange(xmin, xmax + bin_width, bin_width)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    counts = np.zeros(len(bin_centers), dtype=float)
    for xc, yc in zip(x, y):
        if xc < xmin or xc >= xmax:
            continue
        bin_index = int((xc - xmin) // bin_width)
        if bin_index < 0 or bin_index >= len(counts):
            continue
        counts[bin_index] += yc

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        bin_centers, counts, width=bin_width, color="tab:blue", align="center"
    )
    ax.set_xlabel("Deposited Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title(f"{args.title} 14 MeV")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
