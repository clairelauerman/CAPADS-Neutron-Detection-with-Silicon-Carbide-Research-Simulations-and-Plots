# another code iterating through bias voltages with alphas and used as a
# partner code for the Silicon_Scan code to be plotted together
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# finds the directory of the script and adds it to the python import path
# so modules can be imported
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
    conf_text: str,
    bias_v: float,
    distance_mm: float,
    tag: str,
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
    replacement = f'file_name = "Am241alpha_Silicon_Carbide_{distance_int}mm_{bias_int}V.root"'

    print(
        f"SCAN_COMPARISON set_conf function....Replacement ROOT file name is {replacement}"
    )

    # update the name of the root file using replacement
    updated, count = re.subn(
        r'^\s*file_name\s*=\s*".*?\.root"\s*$',
        replacement,
        updated,
        flags=re.MULTILINE,
    )

    if count == 0:
        raise RuntimeError(
            "Could not update ROOTObjectWriter file_name in config"
        )
    # updated is the new config file script that will be returned as conf_text
    return updated


def set_conf_detector_file(conf_text: str, detector_file: str) -> str:
    # Support both detector_file and detectors_file keys.
    updated, count = re.subn(
        r"^\s*detectors?_file\s*=.*$",
        f'detectors_file = "{detector_file}"',
        conf_text,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise RuntimeError("Could not update detectors_file in config")

    # updated is the config script with the correct detector file returned as conf_text
    return updated


# runs the simulation
def run_cmd(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


# read the data from the histograms in the csv files created
def compute_hist_moments_keV(csv_path: Path) -> tuple[float, float, float]:
    print(
        f"SCAN_COMPARISON+++++data being used to compute histogram values is from {csv_path}"
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


def read_deposition_stats(stats_csv: Path) -> tuple[int, int]:
    print(
        f"SCAN_COMPARISON++++++particle and event data is taken from is {stats_csv}"
    )
    with stats_csv.open() as f:
        row = next(csv.DictReader(f))
    total_particles = int(float(row["total_particles"]))
    depositing_particles = int(float(row["depositing_particles"]))
    return total_particles, depositing_particles


def run_bias_scan(
    args, conf_path, source_mac_path, detector_file: str, tag: str
):
    rows = []

    original_conf = conf_path.read_text()
    original_source_mac = source_mac_path.read_text()

    for v in args.bias_voltages_v:
        try:
            print(f"[scan:{tag}] bias={v:g}V detector={detector_file}")
            # replace detector file path in allpix simulation
            conf_text = set_conf_detector_file(original_conf, detector_file)
            # replace bias voltage and distance in allpix simulation
            conf_text = set_conf_bias_and_output(
                conf_text, v, args.distance_mm, tag
            )

            # write allpix config into conf_path using conf_text
            conf_path.write_text(conf_text)
            print(f"[scan:{tag}] wrote detectors_file into {conf_path}")

            # copies the original source mac file into a variable
            source_mac_path.write_text(original_source_mac)
            # runs function to replace the distance in the original with the distance argument from command line
            set_macro_source(source_mac_path, args.distance_mm)

            # goes into allpix output and selects the correct ROOT file
            root_out = (
                Path("/home/claire/allpix-squared/output")
                / f"Am241alpha_Silicon_Carbide_{int(round(args.distance_mm))}mm_{int(round(v))}V.root"
            )

            print(f"SCAN_COMPARISON>>>>>>root_out file is {root_out}")

            # path to the histogram ROOT output file
            hist_root = root_out.with_name(f"{root_out.stem}_histograms.root")

            print(
                f"SCAN_COMPARISON>>>>>>path to histograms within ROOT output file is {hist_root}"
            )

            # path to the csv file written from the ROOT file
            stats_csv = root_out.with_name(f"{root_out.stem}_stats.csv")

            run_cmd(
                [
                    args.run_script,  # path to bash script that runs allpix
                    str(conf_path),  # allpix config file
                    str(args.distance_mm),  # distance arguments
                    str(root_out),  # ROOT output file
                ],
                # cwd is the parent directory of the allpix file converted to a string
                cwd=str(conf_path.parent),
            )

            # stats_csv comes from the fit_collected_charge and contains data on event counts
            # run_csv comes from the extract script and contains histogram distribution data

            run_csv = (
                Path(args.outdir)
                / f"{args.hist_name}_{tag}_{int(round(v))}V.csv"
            )

            # run extract script
            run_cmd(
                [
                    sys.executable,
                    args.extract_script,
                    str(hist_root),
                    args.hist_name,
                    "None",
                    str(run_csv),
                ],
                cwd=str(conf_path.parent),
            )

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
                    sum_e / total_particles
                    if total_particles > 0
                    else float("nan")
                )
                if total_particles > 0:
                    mean2_all = sum_e2 / total_particles
                    var_all = max(0.0, mean2_all - mean_all_keV**2)
                    mean_all_error_keV = np.sqrt(var_all / total_particles)
            except Exception as exc:
                print(
                    f"[scan:{tag}] Could not compute unconditional mean at {v} V: {exc}"
                )

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

        # restore the original source and config files
        finally:
            conf_path.write_text(original_conf)
            source_mac_path.write_text(original_source_mac)

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
        default=10.0,
        help="Fixed source distance in mm for /gps/pos/centre z in source.mac.",
    )

    # path to detector config with the top metal layers
    parser.add_argument(
        "--detector-metal",
        default="/home/claire/allpix-squared/examples/SiC_3x3_detectorset/SiC_detector.conf",
        help="Detector file including top metal and Ti layer.",
    )

    # path to config file
    parser.add_argument(
        "--conf",
        default="/home/claire/allpix-squared/SiC_3x3/SiC_3x3.conf",
        help="Allpix config file path.",
    )

    # path to source.mac file
    parser.add_argument(
        "--source-mac",
        default="/home/claire/allpix-squared/examples/SiC_3x3_detectorset/source.mac",
        help="Geant4 macro containing /gps/pos/centre to set fixed distance.",
    )

    # path to the bash script that runs allpix
    parser.add_argument(
        "--run-script",
        default="/home/claire/allpix-squared/SiC_3x3/run_SiC.sh",
        help="Script that runs Allpix and calls ROOT macro.",
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

    # set directory to csv and plots
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # make variables for all the different scripts
    conf_path = Path(args.conf)
    source_mac_path = Path(args.source_mac)
    detector_metal = Path(args.detector_metal)

    # pass the command line inputs bias voltage and distance, the config path, sourcemac path, detector files, and tag to the function

    rows_on = run_bias_scan(
        args, conf_path, source_mac_path, detector_metal, "ON"
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    rows_sorted = sorted(rows_on, key=lambda r: r["bias_voltage_v"])
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
        label="Metal ON",
        color="tab:blue",
    )

    ax.set_xlabel("Bias voltage [V]")
    ax.set_ylabel("Mean deposited energy [keV]")

    ax.set_title(
        f"3x3 mm2 SiC Bias Voltage vs SiC Mean Deposited Energy (distance={args.distance_mm} mm)"
    )
    ax.grid(True, alpha=0.3)

    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, loc="best")

    fig.tight_layout()
    fig.savefig(outdir / "bias_vs_mean_energy_SiC.png")
    plt.close(fig)

    print(f"Wrote plot: {outdir / 'bias_vs_mean_energy_SiC.png'}")


if __name__ == "__main__":
    main()
