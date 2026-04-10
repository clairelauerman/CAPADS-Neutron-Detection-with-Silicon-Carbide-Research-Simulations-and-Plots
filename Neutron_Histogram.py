#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


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
        default=10000.0,
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

    xmin = args.xmin  # minimum deposited energy plotted
    xmax = args.xmax  # maximum deposited energy plotted
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
            # add up all the counts from the source energy
            counts_matrix[energy_to_idx[energy], bin_index] += yc

    fig, ax = plt.subplots(figsize=(9, 6))
    z = counts_matrix.T
    z = np.where(z > 0, z, np.nan)

    energy_centers = np.asarray(energies_sorted, dtype=float)
    if energy_centers.size == 1:
        half_step = 0.5
        energy_edges = np.array(
            [energy_centers[0] - half_step, energy_centers[0] + half_step]
        )
    else:
        energy_steps = np.diff(energy_centers)
        energy_edges = np.empty(energy_centers.size + 1, dtype=float)
        energy_edges[1:-1] = energy_centers[:-1] + energy_steps / 2.0
        energy_edges[0] = energy_centers[0] - energy_steps[0] / 2.0
        energy_edges[-1] = energy_centers[-1] + energy_steps[-1] / 2.0

    mesh = ax.pcolormesh(
        energy_edges,
        edges,
        z,
        shading="auto",
        norm=LogNorm(),
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Counts (log scale)")

    ax.set_xlabel("Source Energy (MeV)")
    ax.set_ylabel("Deposited Energy (keV)")
    ax.set_title(args.title)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"Wrote plot: {args.out}")


if __name__ == "__main__":
    main()
