# compares the simulated neutron data to the experimental data collected with
# a CHUBUT amplifier at Rez laboratory in Czech Republic
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import Extract_Neutron
import matplotlib.pyplot as plt
import numpy as np

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


def read_ROOT_files(args, macro_number: float):
    rows = []

    for e in args.neutron_energies_MeV:
        energy_label = int(round(e))

        # goes into allpix output and selects the correct ROOT file
        root_file = (
            Path("/home/claire/allpix-squared/output")
            / f"Neutrons_{int(round(args.distance_mm))}mm_{energy_label}MeV.root"
        )
        print(f"PLOT_ROOT>>>>>>root_out file is {root_file}")
        if not root_file.exists():
            print(
                f"PLOT_ROOT WARNING: missing input ROOT file for {energy_label} MeV, skipping."
            )
            continue

        # run MACRO file
        try:
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
        except subprocess.CalledProcessError as exc:
            print(
                f"PLOT_ROOT WARNING: ROOT macro failed for {energy_label} MeV ({exc}), skipping."
            )
            continue

        # path to the histogram ROOT output file
        hist_root = root_file.with_name(f"{root_file.stem}_histograms.root")

        print(
            f"PLOT_ROOT>>>>>>path to histograms within ROOT output file is {hist_root}"
        )

        # stats_csv comes from the fit_collected_charge and contains data on event counts
        # run_csv comes from the extract script and contains histogram distribution data

        # path to the csv file written from the ROOT file
        stats_csv = root_file.with_name(f"{root_file.stem}_stats.csv")

        run_csv = Path(args.outdir) / f"{args.hist_name}_{energy_label}MeV.csv"

        if not hist_root.exists():
            print(
                f"PLOT_ROOT WARNING: histogram ROOT output not found for {energy_label} MeV, skipping."
            )
            continue
        if not stats_csv.exists():
            print(
                f"PLOT_ROOT WARNING: stats CSV not found for {energy_label} MeV, skipping."
            )
            continue

        # run extract script

        root_file_extract = str(hist_root)
        hist_name = args.hist_name
        dir_path = ""
        out_csv = str(run_csv)

        try:
            Extract_Neutron.th1_to_csv(
                root_file_extract, hist_name, dir_path, out_csv
            )
        except Exception as exc:
            print(
                f"PLOT_ROOT WARNING: could not extract histogram for {energy_label} MeV ({exc}), skipping."
            )
            continue

        # intitial data variables
        total_particles = float("nan")
        depositing_particles = float("nan")
        efficiency = float("nan")
        efficiency_error = float("nan")
        mean_all_keV = float("nan")
        mean_all_error_keV = float("nan")

        try:
            total_particles, depositing_particles = read_deposition_stats(
                stats_csv
            )
            sum_e, sum_e2, _ = compute_hist_moments_keV(run_csv)
        except Exception as exc:
            print(
                f"PLOT_ROOT WARNING: could not compute metrics for {energy_label} MeV ({exc}), skipping."
            )
            continue

        efficiency = (
            depositing_particles / total_particles
            if total_particles > 0
            else float("nan")
        )
        if total_particles > 0:
            efficiency_error = np.sqrt(
                efficiency * (1.0 - efficiency) / total_particles
            )
        mean_all_keV = (
            sum_e / total_particles if total_particles > 0 else float("nan")
        )
        if total_particles > 0:
            mean2_all = sum_e2 / total_particles
            var_all = max(0.0, mean2_all - mean_all_keV**2)
            mean_all_error_keV = np.sqrt(var_all / total_particles)

        rows.append(
            {
                "neutron_energy_mev": e,
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


def load_neutron_data(args):

    energies = []
    run_numbers = []

    if not args.bias_voltage_v:
        return energies, run_numbers

    if len(args.bias_voltage_v) > 1:
        print(
            "PLOT_ROOT WARNING: multiple bias voltages provided; using the first value."
        )

    int_v = int(round(args.bias_voltage_v[0]))

    for run_idx in range(3, 10):
        base_dir = (
            f"/home/claire/allpix-squared/NEUTRON_DATA/{int_v}V {run_idx}"
        )

        # iterate through each measurement file
        for meas_idx in range(1, 3):

            num = f"{meas_idx:05d}"

            file_path_1 = os.path.join(base_dir, f"C1meas{num}.txt")

            # vector of time values and amplitude values from the data files
            C1x, C1y = load_file_2col(file_path_1)

            # calculate area under pulse for bias voltage and convert to energy
            energy = calculate_integral(C1x, C1y)

            energies.append(energy)
            run_numbers.append(run_idx)

    return energies, run_numbers


def main():

    # make function take arguemnts for bias voltage and distance from the command line

    parser = argparse.ArgumentParser(
        description="Scan bias voltage at fixed source distance."
    )
    parser.add_argument(
        "--bias-voltage-v",
        nargs="+",
        type=float,
        required=True,
        help="Bias voltages in V the experiment was conducted in.",
    )

    # Energy of the experimental neutrons
    parser.add_argument(
        "--neutron-energies-MeV",
        nargs="+",
        type=float,
        required=True,
        help="Neutron energy(ies) of the experimental beam.",
    )

    # distance of the source from the detector in the simulation
    parser.add_argument(
        "--distance-mm",
        type=float,
        default=30.00,
        help="Fixed source distance in mm for /gps/pos/centre z in source.mac.",
    )

    # path to the extract script that writes the csv files
    parser.add_argument(
        "--extract-script",
        default="/home/claire/allpix-squared/Neutrons/Extract_Neutrons.py",
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
        default="/home/claire/allpix-squared/Neutrons/ROOT_plots",
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
    rows_neutrons = read_ROOT_files(args, 7.83)

    # Experimental
    energies, run_numbers = load_neutron_data(args)

    # sorts the energy by bias voltage
    unique_number = sorted(set(run_numbers))
    mean_energy = []

    # takes the mean of all the energies at the same bias voltage
    for b in unique_number:
        e = [energies[i] for i in range(len(energies)) if run_numbers[i] == b]
        mean_energy.append(np.mean(e))

    fig, ax = plt.subplots(figsize=(7, 5))

    rows_sorted = sorted(rows_neutrons, key=lambda r: r["neutron_energy_mev"])
    x = np.array([r["neutron_energy_mev"] for r in rows_sorted])
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
    exp_energy = 20
    ax.errorbar(
        exp_energy,
        np.mean(mean_energy),
        marker="o",
        linestyle="-",
        markersize=6,
        label="Experimental",
        color="tab:green",
    )

    ax.set_xlabel("Neutron Energy [MeV]")
    ax.set_ylabel("Mean Deposited Energy [keV]")

    ax.set_title(
        f"Neutron Simulation vs Experiment (distance={args.distance_mm} mm)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(outdir / "Neutron_vs_Experiment.png")
    plt.close(fig)

    print(f"Wrote plot: {outdir / 'Neutron_vs_Experiment.png'}")


if __name__ == "__main__":
    main()
