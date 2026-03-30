# Iterates through different diode bias voltages to see how depostied neutron
# energy changes in the SiC detector. Writes ROOT files, extracts the data, fits
# the data, and plots
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# set the source positon in the source.mac file
def set_macro_source(
    source_mac_path: Path, distance_mm: float, e: float
) -> None:
    # Keep source distance fixed while scanning bias voltage.
    original = source_mac_path.read_text()
    # update the position of the source to the command line input
    updated, count_center = re.subn(
        r"^/gps/pos/centre\s+.*$",
        f"/gps/pos/centre {distance_mm:g} 0 0 mm",
        original,
        flags=re.MULTILINE,
    )
    if count_center == 0:
        updated = original.replace(
            "/gps/pos/type Point",
            f"/gps/pos/type Point\n/gps/pos/centre {distance_mm:g} 0 0 mm",
        )

    updated, count_energy = re.subn(
        r"^/gps/ene/mono\s+.*$",
        f"/gps/ene/mono {e:g} MeV",
        updated,
        flags=re.MULTILINE,
    )
    if count_energy == 0:
        updated += f"\n/gps/ene/mono {e:g} MeV\n"

    # set the beam direction of the source
    updated, count_dir = re.subn(
        r"^/gps/direction\s+.*$",
        "/gps/direction -1 0 0",
        updated,
        flags=re.MULTILINE,
    )
    # set mim and max source angles
    if count_dir == 0:
        updated += "\n/gps/direction -1 0 0\n"

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
def set_conf_distance_and_output(
    original_conf: str,
    energy_mev: float,
    distance_mm: float,
) -> str:
    updated = re.sub(
        r"^\s*source_position\s*=\s*.*\{DISTANCE\}.*$",
        "source_position = 0um 0um 0um",
        original_conf,
        flags=re.MULTILINE,
    )

    # make distance and bias voltage integers
    distance_int = int(round(distance_mm))
    energy_int = int(round(energy_mev))

    # create the correct name of root file to be written
    replacement = (
        f'file_name = "Neutrons_{distance_int}mm_{energy_int}MeV.root"'
    )

    print(
        f"ROOT_EDITOR set_conf function....Replacement ROOT file name is {replacement}"
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


def load_hist_xy(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return np.array([]), np.array([])
    if {"x", "content"}.issubset(data.dtype.names):
        return np.array(data["x"]), np.array(data["content"])
    if {"bin_center", "count"}.issubset(data.dtype.names):
        return np.array(data["bin_center"]), np.array(data["count"])
    raise RuntimeError(
        f"Unexpected CSV headers in {csv_path}. Expected x/content or bin_center/count."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Scan neutron energy at fixed source distance (bias held constant in config)."
    )
    parser.add_argument(
        "--neutron-energies-MeV",
        nargs="+",
        type=float,
        required=True,
        help="Neutron energies in MeV.",
    )
    parser.add_argument(
        "--distance-mm",
        type=float,
        default=30.0,
        help="Fixed source distance in mm for /gps/pos/centre z in source.mac.",
    )
    parser.add_argument(
        "--conf",
        default="/home/claire/allpix-squared/Neutrons/Neutrons.conf",
        help="Allpix config file path.",
    )
    parser.add_argument(
        "--source-mac",
        default="/home/claire/allpix-squared/Neutrons/neutron_source.mac",
        help="Geant4 macro containing /gps/pos/centre to set fixed distance.",
    )
    parser.add_argument(
        "--run-script",
        default="/home/claire/allpix-squared/Neutrons/run_neutrons.sh",
        help="Script that runs Allpix and calls ROOT macro.",
    )
    parser.add_argument(
        "--extract-script",
        default="/home/claire/allpix-squared/Neutrons/Extract_Neutron.py",
        help="CSV extractor script path.",
    )
    parser.add_argument(
        "--hist-name",
        default="h_deposited_energy",
        help="Histogram name to extract and fit.",
    )
    parser.add_argument(
        "--outdir",
        default="/home/claire/allpix-squared/Neutrons/ROOT_plots",
        help="Output directory for CSV and plots.",
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
        for e in args.neutron_energies_MeV:
            conf_path.write_text(original_conf)
            source_mac_path.write_text(original_source_mac)

            set_macro_source(source_mac_path, args.distance_mm, e)
            conf_path.write_text(
                set_conf_distance_and_output(
                    original_conf, e, args.distance_mm
                )
            )

            root_out = (
                Path("/home/claire/allpix-squared/output")
                / f"Neutrons_{int(round(args.distance_mm))}mm_{int(round(e))}MeV.root"
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

            energy_label = int(round(e))
            hist_root = root_out.with_name(f"{root_out.stem}_histograms.root")
            stats_csv = root_out.with_name(f"{root_out.stem}_stats.csv")
            run_csv = outdir / f"{args.hist_name}_{energy_label}MeV.csv"

            total_particles = float("nan")
            depositing_particles = float("nan")
            efficiency = float("nan")
            efficiency_error = float("nan")
            mean_all_keV = float("nan")
            mean_all_error_keV = float("nan")

            if not hist_root.exists() or not stats_csv.exists():
                print(
                    f"Skipping {energy_label} MeV: missing analysis outputs "
                    f"(hist={hist_root.exists()}, stats={stats_csv.exists()})"
                )
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
                continue

            try:
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
            except subprocess.CalledProcessError as exc:
                print(
                    f"Skipping {energy_label} MeV: histogram extraction failed ({exc})"
                )
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
                continue

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
                print(
                    f"Could not compute unconditional mean at {e} MeV: {exc}"
                )

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

    # Overlay all per-energy histograms on a single plot
    rows_sorted = sorted(rows, key=lambda r: r["neutron_energy_mev"])
    fig, ax = plt.subplots(figsize=(9, 6))
    n_plotted = 0
    for r in rows_sorted:
        csv_path = Path(r["hist_csv"])
        if not csv_path.exists():
            continue
        try:
            x, y = load_hist_xy(csv_path)
        except Exception as exc:
            print(
                f"Skipping histogram overlay for {r['neutron_energy_mev']} MeV: {exc}"
            )
            continue
        if x.size == 0 or y.size == 0:
            continue
        energy = int(round(r["neutron_energy_mev"]))
        ax.plot(x, y, linewidth=1.2, label=f"{energy} MeV")
        n_plotted += 1

    if n_plotted == 0:
        raise RuntimeError("No histogram CSV files available to plot.")

    ax.set_xlim(0, 5000)
    ax.set_xlabel("Deposited Energy [keV]")
    ax.set_ylabel("Counts")
    ax.set_title(
        f"Deposited Energy Histograms Overlay (distance = {args.distance_mm:g} mm)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "Neutron_hist_overlay.png")
    plt.close(fig)

    # confirmation print statements
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote plot: {outdir / 'Neutron_hist_overlay.png'}")


if __name__ == "__main__":
    main()
