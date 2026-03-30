# compares the simulated data for alphas hitting a SiC detector to the experimental
# data collected with a CIVIDEC amplifier at REZ laboratory
import argparse
import csv
import os
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


def calculate_integral(time, amplitude):
    if len(time) < 2 or len(amplitude) < 2:
        return float("nan")
    # sort by time to avoid negative/garbled integration when timestamps
    # are unsorted or wrapped
    order = np.argsort(time)
    time = np.asarray(time)[order]
    amplitude = np.asarray(amplitude)[order]
    baseline_window = min(50, len(amplitude))
    baseline = np.mean(amplitude[:baseline_window])
    amplitude = amplitude - baseline
    pulse_area = abs(np.trapezoid(amplitude, time))
    amplifier_gain = 40  # db
    G = 10 ** (amplifier_gain / 20)
    charge = pulse_area / G / 50
    energy = charge * 7.83 / (1.602 * (10**-19)) / 1000

    return energy


def load_file_2col(file_path, colX=0, colY=1):
    x_values = []
    y_values = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) > max(colX, colY):
                try:
                    x = float(parts[colX])
                    y = float(parts[colY])
                    x_values.append(x)
                    y_values.append(y)
                except ValueError:
                    continue

    return x_values, y_values


def load_alpha_data(args):

    energies = []
    bias_values = []

    for v in args.bias_voltages_v:
        int_v = int(round(v))
        base_dir = f"/home/claire/allpix-squared/ALPHA_DATA/{int_v}V"

        # iterate through each measurement file
        for i in range(1, 250):

            num = f"{i:05d}"

            file_path_1 = os.path.join(base_dir, f"C1meas{num}.txt")

            if not os.path.exists(file_path_1):
                continue

            # vector of time values and amplitude values from the data files
            C1x, C1y = load_file_2col(file_path_1)
            if len(C1x) < 2 or len(C1y) < 2:
                continue

            # calculate area under pulse for bias voltage and convert to energy
            energy = calculate_integral(C1x, C1y)
            if not np.isfinite(energy):
                continue
            energies.append(energy)
            bias_values.append(v)

    return energies, bias_values


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
        default=13.75,
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

    # Simulation
    rows_Silicon_Carbide = read_ROOT_files(args, "Silicon_Carbide", 7.83)

    # Experimental
    energies, bias_values = load_alpha_data(args)

    # sorts the energy by bias voltage
    unique_bias = sorted(set(bias_values))
    mean_energy = []

    # takes the mean of all the energies at the same bias voltage
    for b in unique_bias:
        e = [energies[i] for i in range(len(energies)) if bias_values[i] == b]
        mean_energy.append(np.mean(e))

    fig, ax = plt.subplots(figsize=(7, 5))

    # plot Simulation
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
        label="Simulation",
        color="tab:red",
    )

    # plot experiment
    ax.errorbar(
        unique_bias,
        mean_energy,
        marker="o",
        linestyle="-",
        markersize=6,
        label="Experimental",
        color="tab:green",
    )

    ax.set_xlabel("Bias Voltage [V]")
    ax.set_ylabel("Mean Deposited Energy [keV]")

    ax.set_title("Alpha Simulation vs Experimental Channel 1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(outdir / "Sim_vs_Exp_Alpha.png")
    plt.close(fig)

    print(f"Wrote plot: {outdir / 'Sim_vs_Exp_Alpha.png'}")


if __name__ == "__main__":
    main()
