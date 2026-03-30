import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SiC_Extract

# finds the directory of the script and adds it to the python import path
# so modules can be imported
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# reads the csv file from the macro scipt that contains particle and event data
def read_deposition_stats(stats_csv: Path) -> tuple[int, int]:
    print(
        f"PLOT_ROOT++++++particle and event data is taken from is {stats_csv}"
    )
    with stats_csv.open() as f:
        row = next(csv.DictReader(f))
    total_particles = int(float(row["total_particles"]))
    depositing_particles = int(float(row["depositing_particles"]))
    return total_particles, depositing_particles


# read the data from the histograms in the csv files created
def compute_hist_moments_keV(csv_path: Path) -> tuple[float, float, float]:
    print(
        f"PLOT_ROOT+++++data being used to compute histogram values is from {csv_path}"
    )
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return 0.0, 0.0, 0.0
    if {"x", "content"}.issubset(data.dtype.names):
        x = data["x"]
        c = data["content"]
        return (
            float(np.sum(x * c)),
            float(np.sum((x**2) * c)),
            float(np.sum(c)),
        )
    if {"bin_center", "count"}.issubset(data.dtype.names):
        x = data["bin_center"]
        c = data["count"]
        return (
            float(np.sum(x * c)),
            float(np.sum((x**2) * c)),
            float(np.sum(c)),
        )
    raise RuntimeError(
        f"Unexpected CSV headers in {csv_path}. Expected x/content or bin_center/count."
    )


def read_ROOT_files(args, tag: str, macro_number: float):
    rows = []

    for v in args.bias_voltages_v:

        # goes into allpix output and selects the correct ROOT file
        root_file = (
            Path("/home/claire/allpix-squared/output")
            / f"Am241alpha_{tag}_{int(round(args.distance_mm))}mm_{int(round(v))}V.root"
        )
        print(f"PLOT_ROOT>>>>>>root_out file is {root_file}")

        # run MACRO file
        subprocess.run(
            [
                "root",
                "-l",
                "-b",
                "-q",
                f'/home/claire/allpix-squared/SiC_3x3/fit_collected_charge.C("{root_file}", {macro_number})',
            ],
            check=True,
        )

        # path to the histogram ROOT output file
        hist_root = root_file.with_name(f"{root_file.stem}_histograms.root")

        print(
            f"PLOT_ROOT>>>>>>path to histograms within ROOT output file is {hist_root}"
        )

        # stats_csv comes from the fit_collected_charge and contains data on event counts
        # run_csv comes from the extract script and contains histogram distribution data

        # path to the csv file written from the ROOT file
        stats_csv = root_file.with_name(f"{root_file.stem}_stats.csv")

        run_csv = (
            Path(args.outdir) / f"{args.hist_name}_{tag}_{int(round(v))}V.csv"
        )

        # run extract script

        root_file_extract = str(hist_root)
        hist_name = args.hist_name
        dir_path = ""
        out_csv = str(run_csv)

        SiC_Extract.th1_to_csv(root_file_extract, hist_name, dir_path, out_csv)

        # intitial data variables
        total_particles = float("nan")
        depositing_particles = float("nan")
        efficiency = float("nan")
        efficiency_error = float("nan")
        mean_all_keV = float("nan")
        mean_all_error_keV = float("nan")

        total_particles, depositing_particles = read_deposition_stats(
            stats_csv
        )

        efficiency = (
            depositing_particles / total_particles
            if total_particles > 0
            else float("nan")
        )
        if total_particles > 0:
            efficiency_error = np.sqrt(
                efficiency * (1.0 - efficiency) / total_particles
            )
        sum_e, sum_e2, _ = compute_hist_moments_keV(run_csv)
        mean_all_keV = (
            sum_e / total_particles if total_particles > 0 else float("nan")
        )
        if total_particles > 0:
            mean2_all = sum_e2 / total_particles
            var_all = max(0.0, mean2_all - mean_all_keV**2)
            mean_all_error_keV = np.sqrt(var_all / total_particles)

        rows.append(
            {
                "bias_voltage_v": v,
                "distance_mm": args.distance_mm,
                "mean_all_keV": mean_all_keV,
                "mean_all_error_keV": mean_all_error_keV,
                "efficiency": efficiency,
                "efficiency_error": efficiency_error,
                "total_particles": total_particles,
                "depositing_particles": depositing_particles,
                "hist_csv": str(run_csv),
            }
        )

    return rows


def main():

    # make function take arguemnts for bias voltage and distance from the command line

    parser = argparse.ArgumentParser(
        description="Scan bias voltage at fixed source distance."
    )
    parser.add_argument(
        "--bias-voltages-v",
        nargs="+",
        type=float,
        required=True,
        help="Bias voltages in V for ElectricFieldReader.bias_voltage.",
    )
    parser.add_argument(
        "--distance-mm",
        type=float,
        default=30.0,
        help="Fixed source distance in mm for /gps/pos/centre z in source.mac.",
    )

    # path to the extract script that writes the csv files
    parser.add_argument(
        "--extract-script",
        default="/home/claire/allpix-squared/SiC_3x3/SiC_Extract.py",
        help="CSV extractor script path.",
    )

    # path to the root histogram files created in root
    parser.add_argument(
        "--hist-root",
        default="/home/claire/allpix-squared/deposited_histograms.root",
        help="Histogram ROOT file created by the run script.",
    )

    # name of the histograms
    parser.add_argument(
        "--hist-name",
        default="h_deposited_energy",
        help="Histogram name to extract and fit.",
    )

    # path to the directory where csv files and plots are stored
    parser.add_argument(
        "--outdir",
        default="/home/claire/allpix-squared/SiC_3x3/bias_scan",
        help="Output directory for CSV and plots.",
    )

    # path to the csv file that stores the energy values converted from the charge histograms
    parser.add_argument(
        "--stats-csv",
        default="/home/claire/allpix-squared/deposited_stats.csv",
        help="Stats CSV created by fit_deposited_charge.C",
    )

    args = parser.parse_args()

    # makes a directory to store the plots in
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # rows_Silicon = read_ROOT_files(args, "Silicon", 3.64)

    rows_Silicon_Carbide = read_ROOT_files(args, "Silicon_Carbide", 7.83)

    fig, ax = plt.subplots(figsize=(7, 5))

    rows_sorted = sorted(
        rows_Silicon_Carbide, key=lambda r: r["bias_voltage_v"]
    )
    x = np.array([r["bias_voltage_v"] for r in rows_sorted])
    y = np.array([r["mean_all_keV"] for r in rows_sorted])
    y_err = np.array([r["mean_all_error_keV"] for r in rows_sorted])
    ax.errorbar(
        x,
        y,
        yerr=y_err,
        fmt="o-",
        markersize=6,
        capsize=4,
        label="Alpha",
        color="tab:red",
    )

    ax.set_xlabel("Bias Voltage [V]")
    ax.set_ylabel("Mean Deposited Energy [keV]")

    ax.set_title(
        f"Alpha Bias Voltage vs Mean Deposited Energy (distance={args.distance_mm} mm)"
    )
    ax.grid(True, alpha=0.3)

    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, loc="best")

    fig.tight_layout()
    fig.savefig(outdir / "bias_vs_mean_energy_Alpha.png")
    plt.close(fig)

    print(f"Wrote plot: {outdir / 'bias_vs_mean_energy_Alpha.png'}")


if __name__ == "__main__":
    main()
