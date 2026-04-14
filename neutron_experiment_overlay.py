#!/usr/bin/env python3
# Overlay simulation histogram (CSV) with experimental histogram for one energy.
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# find script directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from neutron_experiment_histogram import load_neutron_data  # noqa: E402


def infer_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    if centers.size < 2:
        half_step = 0.5
        return np.array([centers[0] - half_step, centers[0] + half_step])
    diffs = np.diff(centers)
    step = float(np.median(diffs))
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = centers[:-1] + step / 2.0
    edges[0] = centers[0] - step / 2.0
    edges[-1] = centers[-1] + step / 2.0
    return edges


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
        description="Overlay simulation and experimental deposited-energy histograms for one energy."
    )
    parser.add_argument(
        "--energy",
        type=float,
        required=True,
        help="Neutron source energy in MeV (e.g., 15.7, 17, 20).",
    )
    parser.add_argument(
        "--sim-csv",
        required=True,
        help="Path to simulation histogram CSV (e.g., h_deposited_energy_20MeV.csv).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (defaults to <outdir>/overlay_<energy>MeV.png).",
    )
    parser.add_argument(
        "--outdir",
        default="/home/claire/allpix-squared/Neutrons/ROOT_plots",
        help="Output directory for plots.",
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
        default=None,
        help="Maximum deposited energy (keV) to include (default: auto from sim).",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=None,
        help="Histogram bin width in keV (default: inferred from sim CSV).",
    )
    parser.add_argument(
        "--title",
        default="Deposited Energy: Simulation vs Experiment",
        help="Plot title.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize both histograms to unit area.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use log scale on the y-axis.",
    )
    parser.add_argument(
        "--bias-voltage-v",
        nargs="+",
        type=float,
        default=300,
        help="Bias voltage for experiment (used for 20 MeV REZ data).",
    )

    args = parser.parse_args()

    sim_csv = Path(args.sim_csv)
    if not sim_csv.exists():
        raise RuntimeError(f"Simulation CSV not found: {sim_csv}")

    # Load simulation histogram (already binned)
    sim_x, sim_counts = read_hist_csv(sim_csv)
    sim_x = np.asarray(sim_x, dtype=float)
    sim_counts = np.asarray(sim_counts, dtype=float)

    if args.bin_width is None:
        sim_edges = infer_edges_from_centers(sim_x)
        bin_width = float(np.median(np.diff(sim_edges)))
    else:
        bin_width = args.bin_width
        if bin_width <= 0:
            raise RuntimeError("--bin-width must be > 0")
        sim_edges = np.arange(
            sim_x.min() - bin_width / 2, sim_x.max() + bin_width, bin_width
        )

    xmin = args.xmin
    if args.xmax is None:
        xmax = float(sim_edges[-1])
    else:
        xmax = args.xmax
    if xmax <= xmin:
        raise RuntimeError("--xmax must be greater than --xmin")

    # Build experimental histogram using same binning
    exp_samples = load_neutron_data(args, args.energy)
    exp_samples = np.asarray(exp_samples, dtype=float)
    exp_samples = exp_samples[np.isfinite(exp_samples)]

    exp_counts, exp_edges = np.histogram(
        exp_samples, bins=sim_edges, range=(xmin, xmax)
    )
    exp_centers = (exp_edges[:-1] + exp_edges[1:]) / 2.0

    # Apply range mask to simulation and experiment
    in_range_mask = (sim_x >= xmin) & (sim_x < xmax)
    sim_x_plot = sim_x[in_range_mask]
    sim_counts_plot = sim_counts[in_range_mask]

    exp_mask = (exp_centers >= xmin) & (exp_centers < xmax)
    exp_centers_plot = exp_centers[exp_mask]
    exp_counts_plot = exp_counts[exp_mask]

    if args.normalize:
        sim_area = np.sum(sim_counts_plot)
        exp_area = np.sum(exp_counts_plot)
        if sim_area > 0:
            sim_counts_plot = sim_counts_plot / sim_area
        if exp_area > 0:
            exp_counts_plot = exp_counts_plot / exp_area

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(
        sim_x_plot,
        sim_counts_plot,
        where="mid",
        label="Simulation",
        color="tab:blue",
    )
    ax.step(
        exp_centers_plot,
        exp_counts_plot,
        where="mid",
        label="Experiment",
        color="tab:orange",
    )

    ax.set_xlabel("Deposited Energy (keV)")
    ax.set_ylabel("Counts" if not args.normalize else "Normalized Counts")
    ax.set_title(f"{args.title} {args.energy:g} MeV")
    ax.legend()

    if args.logy:
        ax.set_yscale("log")

    fig.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else outdir / f"overlay_{args.energy:g}MeV.png"
    )
    fig.savefig(out_path, dpi=160)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
