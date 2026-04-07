#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# get the neutron source energy from the name of the csv file
def parse_energy_from_name(path: Path):
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)MeV", path.name)
    if not match:
        return None
    return float(match.group(1))


def read_hist_csv(csv_path: Path):
    # import the data from the csv file
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    # store the data from the csv files in the x and y vectors
    cols = set(data.dtype.names or [])
    if {"x", "content"}.issubset(cols):
        x = data["x"]
        y = data["content"]
        return x, y
    if {"bin_center", "count"}.issubset(cols):
        x = data["bin_center"]
        y = data["count"]
        return x, y
    raise RuntimeError(
        f"Unexpected CSV headers in {csv_path}. Expected x/content or bin_center/count."
    )


def main():
    parser = argparse.ArgumentParser(
        description="2D heatmap from CSVs: source energy vs deposited energy vs counts."
    )
    parser.add_argument(
        "--csv-dir",
        default="/home/claire/allpix-squared/Neutrons/ROOT_plots",
        help="Directory containing histogram CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="h_deposited_energy_*MeV.csv",
        help="Glob pattern for CSV files.",
    )
    parser.add_argument(
        "--out",
        default="/home/claire/allpix-squared/Neutrons/ROOT_plots/deposited_energy_heatmap.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--min-energy",
        type=float,
        default=None,
        help="Minimum neutron energy (MeV) to include.",
    )
    parser.add_argument(
        "--max-energy",
        type=float,
        default=None,
        help="Maximum neutron energy (MeV) to include.",
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
        default=2000.0,
        help="Maximum deposited energy (keV) to include.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=100.0,
        help="Deposited energy bin width in keV (default: 1000).",
    )
    parser.add_argument(
        "--logz",
        action="store_true",
        help="Use log scale on counts (color scale).",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Draw grid lines at bin boundaries (off by default).",
    )
    parser.add_argument(
        "--title",
        default="Heatmap: Source Energy vs Deposited Energy vs Counts",
        help="Plot title.",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    # put all the csv files into a list
    files = sorted(csv_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(
            f"No CSV files found in {csv_dir} matching {args.pattern}"
        )

    items = []
    # from each file get the neutron source energy, x data, and y data
    for p in files:
        energy = parse_energy_from_name(p)
        if energy is None:
            continue
        if args.min_energy is not None and energy < args.min_energy:
            continue
        if args.max_energy is not None and energy > args.max_energy:
            continue
        x, y = read_hist_csv(p)
        items.append((energy, x, y))

    if not items:
        raise RuntimeError("No CSVs matched the energy filters.")

    items.sort(key=lambda t: t[0])

    xmin = args.xmin  # minimum neutron source energy plotted
    xmax = args.xmax  # maximum neutron source energy plotted
    bin_width = args.bin_width  # size of the grid
    edges = np.arange(xmin, xmax + bin_width, bin_width)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    # create a matrix of the energies
    energies = [e for e, _, _ in items]
    energies_sorted = sorted(energies)
    energy_to_idx = {e: i for i, e in enumerate(energies_sorted)}

    # create a matrix of the counts per energy
    counts_matrix = np.zeros(
        (len(energies_sorted), len(bin_centers)), dtype=float
    )

    for energy, x, y in items:
        for xc, yc in zip(x, y):
            if xc < xmin or xc >= xmax:
                continue
            bin_index = int((xc - xmin) // bin_width)
            counts_matrix[energy_to_idx[energy], bin_index] += yc

    fig, ax = plt.subplots(figsize=(9, 6))
    # create the z vector
    z = counts_matrix.T
    if args.logz:
        z = np.where(z > 0, z, np.nan)
        im = ax.imshow(
            np.log10(z),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[energies_sorted[0], energies_sorted[-1], xmin, xmax],
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("log10(Counts)")
    else:
        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[energies_sorted[0], energies_sorted[-1], xmin, xmax],
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Counts")

    ax.set_xlabel("Source Energy (MeV)")
    ax.set_ylabel("Deposited Energy (keV)")
    ax.set_title(args.title)
    if args.grid:
        # Grid lines to show deposited-energy bin boundaries
        y_edges = np.arange(xmin, xmax + bin_width, bin_width)
        for y in y_edges:
            ax.axhline(y, color="white", linewidth=0.5, alpha=0.5)
        # Grid lines to show source-energy steps
        if len(energies_sorted) > 1:
            x_edges = [
                (energies_sorted[i] + energies_sorted[i + 1]) / 2.0
                for i in range(len(energies_sorted) - 1)
            ]
            for x in x_edges:
                ax.axvline(x, color="white", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"Wrote plot: {args.out}")


if __name__ == "__main__":
    main()
