# iterates through different diode bias voltages to see how the deposited energy
# of alpha particles into the SiC detector changes
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# import histogram created in guassian code
from SiC_fit_gaussian import fit_histogram_csv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# set the source positon in the source.mac file
def set_macro_source(source_mac_path: Path, distance_mm: float) -> None:
    # Keep source distance fixed while scanning bias voltage.
    original = source_mac_path.read_text()
    # update the position of the source to the command line input
    updated, count_center = re.subn(
        r"^/gps/pos/centre\s+.*$",
        f"/gps/pos/centre 0 0 {distance_mm:g} mm",
        original,
        flags=re.MULTILINE,
    )
    if count_center == 0:
        updated = original.replace(
            "/gps/pos/type Point",
            f"/gps/pos/type Point\n/gps/pos/centre 0 0 {distance_mm:g} mm",
        )
    # set the beam direction of the source
    updated, count_dir = re.subn(
        r"^/gps/direction\s+.*$",
        "/gps/direction 0 0 -1",
        updated,
        flags=re.MULTILINE,
    )
    # set mim and max source angles
    if count_dir == 0:
        updated += "\n/gps/direction 0 0 -1\n"
    updated, count_theta_min = re.subn(
        r"^/gps/ang/mintheta\s+.*$",
        "/gps/ang/mintheta 0 deg",
        updated,
        flags=re.MULTILINE,
    )
    if count_theta_min == 0:
        updated += "\n/gps/ang/mintheta 0 deg\n"
    updated, count_theta_max = re.subn(
        r"^/gps/ang/maxtheta\s+.*$",
        "/gps/ang/maxtheta 3 deg",
        updated,
        flags=re.MULTILINE,
    )
    if count_theta_max == 0:
        updated += "/gps/ang/maxtheta 3 deg\n"
    source_mac_path.write_text(updated)


# set the source positon to 0 in the main config bc it will be overwritten
# by the source.mac file
def set_conf_bias_and_output(
    conf_text: str, bias_v: float, distance_mm: float
) -> str:
    updated = re.sub(
        r"^\s*source_position\s*=\s*.*\{DISTANCE\}.*$",
        "source_position = 0um 0um 0um",
        conf_text,
        flags=re.MULTILINE,
    )
    # set the bias voltage to command line input
    updated, count_bias = re.subn(
        r"^\s*bias_voltage\s*=.*$",
        f"bias_voltage = {bias_v:g}V",
        updated,
        flags=re.MULTILINE,
    )
    if count_bias == 0:
        raise RuntimeError("Could not update bias_voltage in config")

    # make distance and bias voltage integers
    distance_int = int(round(distance_mm))
    bias_int = int(round(bias_v))
    # create the correct name of root file to be written
    replacement = (
        f'file_name = "Am241alpha_SiC_{distance_int}mm_{bias_int}V.root"'
    )
    updated, count = re.subn(
        r'^\s*file_name\s*=\s*"Am241alpha_SiC_.*?\.root"\s*$',
        replacement,
        updated,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise RuntimeError(
            "Could not update ROOTObjectWriter file_name in config"
        )
    return updated


# runs the simulation
def run_cmd(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


# read the data from the histograms in the csv files created
def compute_hist_moments_keV(csv_path: Path) -> tuple[float, float, float]:
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


def read_deposition_stats(stats_csv: Path) -> tuple[int, int]:
    with stats_csv.open() as f:
        row = next(csv.DictReader(f))
    total_particles = int(float(row["total_particles"]))
    depositing_particles = int(float(row["depositing_particles"]))
    return total_particles, depositing_particles


def main():
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
    parser.add_argument(
        "--conf",
        default="/home/claire/allpix-squared/SiC_3x3/SiC_3x3.conf",
        help="Allpix config file path.",
    )
    parser.add_argument(
        "--source-mac",
        default="/home/claire/allpix-squared/examples/SiC_3x3_detectorset/source.mac",
        help="Geant4 macro containing /gps/pos/centre to set fixed distance.",
    )
    parser.add_argument(
        "--run-script",
        default="/home/claire/allpix-squared/SiC_3x3/run_SiC.sh",
        help="Script that runs Allpix and calls ROOT macro.",
    )
    parser.add_argument(
        "--extract-script",
        default="/home/claire/allpix-squared/SiC_3x3/SiC_Extract.py",
        help="CSV extractor script path.",
    )
    parser.add_argument(
        "--hist-root",
        default="/home/claire/allpix-squared/deposited_histograms.root",
        help="Histogram ROOT file created by the run script.",
    )
    parser.add_argument(
        "--hist-name",
        default="h_deposited_energy",
        help="Histogram name to extract and fit.",
    )
    parser.add_argument(
        "--outdir",
        default="/home/claire/allpix-squared/SiC_3x3/bias_scan",
        help="Output directory for CSV and plots.",
    )
    parser.add_argument(
        "--stats-csv",
        default="/home/claire/allpix-squared/deposited_stats.csv",
        help="Stats CSV created by fit_deposited_charge.C",
    )
    args = parser.parse_args()

    conf_path = Path(args.conf)
    source_mac_path = Path(args.source_mac)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    original_conf = conf_path.read_text()
    original_source_mac = source_mac_path.read_text()
    rows = []

    # iterate through the input bias voltages
    try:
        for v in args.bias_voltages_v:
            conf_path.write_text(original_conf)
            source_mac_path.write_text(original_source_mac)

            set_macro_source(source_mac_path, args.distance_mm)
            conf_path.write_text(
                set_conf_bias_and_output(original_conf, v, args.distance_mm)
            )

            root_out = (
                Path("/home/claire/allpix-squared/output")
                / f"Am241alpha_SiC_{int(round(args.distance_mm))}mm_{int(round(v))}V.root"
            )
            run_cmd(
                [
                    args.run_script,
                    str(conf_path),
                    str(args.distance_mm),
                    str(root_out),
                ],
                cwd=str(conf_path.parent),
            )

            run_csv = outdir / f"{args.hist_name}_{int(round(v))}V.csv"
            run_cmd(
                [
                    sys.executable,
                    args.extract_script,
                    args.hist_root,
                    args.hist_name,
                    "None",
                    str(run_csv),
                ],
                cwd=str(conf_path.parent),
            )

            total_particles = float("nan")
            depositing_particles = float("nan")
            efficiency = float("nan")
            efficiency_error = float("nan")
            mean_all_keV = float("nan")
            mean_all_error_keV = float("nan")
            sigma_fit_keV = float("nan")
            sigma_fit_error_keV = float("nan")

            try:
                total_particles, depositing_particles = read_deposition_stats(
                    Path(args.stats_csv)
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
                sum_energy_keV, sum_energy2_keV2, _ = compute_hist_moments_keV(
                    run_csv
                )
                mean_all_keV = (
                    sum_energy_keV / total_particles
                    if total_particles > 0
                    else float("nan")
                )
                if total_particles > 0:
                    mean2_all = sum_energy2_keV2 / total_particles
                    var_all = max(0.0, mean2_all - mean_all_keV**2)
                    mean_all_error_keV = np.sqrt(var_all / total_particles)
            except Exception as exc:
                print(f"Could not compute unconditional mean at {v} V: {exc}")

            try:
                fit = fit_histogram_csv(str(run_csv))
                sigma_fit_keV = fit["sigma"]
                sigma_fit_error_keV = fit["sigma_error"]
                if (
                    (not np.isfinite(sigma_fit_keV))
                    or (not np.isfinite(sigma_fit_error_keV))
                    or sigma_fit_keV <= 0
                    or (sigma_fit_error_keV / sigma_fit_keV > 0.5)
                ):
                    sigma_fit_keV = float("nan")
                    sigma_fit_error_keV = float("nan")
            except Exception as exc:
                print(f"Could not compute Gaussian sigma at {v} V: {exc}")

            rows.append(
                {
                    "bias_voltage_v": v,
                    "distance_mm": args.distance_mm,
                    "mean_all_keV": mean_all_keV,
                    "mean_all_error_keV": mean_all_error_keV,
                    "sigma_fit_keV": sigma_fit_keV,
                    "sigma_fit_error_keV": sigma_fit_error_keV,
                    "efficiency": efficiency,
                    "efficiency_error": efficiency_error,
                    "total_particles": total_particles,
                    "depositing_particles": depositing_particles,
                    "hist_csv": str(run_csv),
                }
            )

    finally:
        conf_path.write_text(original_conf)
        source_mac_path.write_text(original_source_mac)

    if not rows:
        raise RuntimeError(
            "No bias points were processed; nothing to summarize."
        )

    summary_csv = outdir / "bias_scan_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # make vectors of all the variables to be plotted
    rows_sorted = sorted(rows, key=lambda r: r["bias_voltage_v"])
    # bias voltage values
    x = np.array([r["bias_voltage_v"] for r in rows_sorted])
    # mean deposited energy values
    y_all = np.array([r["mean_all_keV"] for r in rows_sorted])
    # errors for mean deposited energies
    y_all_err = np.array([r["mean_all_error_keV"] for r in rows_sorted])
    # guassian sigma values
    y_sigma = np.array([r["sigma_fit_keV"] for r in rows_sorted])
    # gaussian sigma errors
    y_sigma_err = np.array([r["sigma_fit_error_keV"] for r in rows_sorted])
    # efficiency values
    y_eff = np.array([r["efficiency"] for r in rows_sorted])
    # efficiency errors
    y_eff_err = np.array([r["efficiency_error"] for r in rows_sorted])

    # plot mean deposited energy vs bias voltage
    plt.figure(figsize=(7, 5))
    ax1 = plt.gca()
    h1 = ax1.errorbar(
        x,
        y_all,
        yerr=y_all_err,
        fmt="o-",
        markersize=6,
        capsize=4,
        label="Mean (all emitted)",
        color="tab:blue",
    )
    ax1.set_xlabel("Bias voltage [V]")
    ax1.set_ylabel("Mean deposited energy [keV]", color="tab:blue")
    ax1.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    # add gaussian sigma onto plot
    ax2 = ax1.twinx()
    valid_sigma = np.isfinite(y_sigma) & np.isfinite(y_sigma_err)
    if np.any(valid_sigma):
        h2 = ax2.errorbar(
            x[valid_sigma],
            y_sigma[valid_sigma],
            yerr=y_sigma_err[valid_sigma],
            fmt="s--",
            markersize=5,
            capsize=4,
            label="Gaussian sigma (fit)",
            color="tab:red",
        )
        ymax = np.nanpercentile(
            y_sigma[valid_sigma] + y_sigma_err[valid_sigma], 95
        )
        ax2.set_ylim(bottom=0, top=max(1.0, 1.2 * ymax))
    else:
        h2 = ax2.plot(
            [],
            [],
            "s--",
            color="tab:red",
            label="Gaussian sigma (fit)",
        )[0]
    ax2.set_ylabel("Gaussian sigma [keV]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title(
        f"3x3 mm2 SiC Bias Voltage vs Mean Deposited Energy (distance = {args.distance_mm:g} mm)"
    )
    ax1.legend([h1, h2], ["Mean (all emitted)", "Gaussian sigma (fit)"])
    plt.tight_layout()
    plt.savefig(outdir / "bias_vs_mean_energy.png")
    plt.close()

    # plot of efficiency vs bias voltage
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        x,
        y_eff,
        yerr=y_eff_err,
        fmt="d-",
        markersize=6,
        capsize=4,
    )
    plt.ylim(0, 1.05)
    plt.xlabel("Bias voltage [V]")
    plt.ylabel("Depositing fraction")
    plt.title(
        f"Bias Voltage vs Deposition Efficiency (distance = {args.distance_mm:g} mm)"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "bias_vs_efficiency.png")
    plt.close()

    # confirmation print statements
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote plot: {outdir / 'bias_vs_mean_energy.png'}")
    print(f"Wrote plot: {outdir / 'bias_vs_efficiency.png'}")


if __name__ == "__main__":
    main()
