# compares the simulated neutron data to the experimental data collected with
# a CIVIDEC amplifier at Rez laboratory in Czech Republic
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# finds the directory of the script and adds it to the python import path
# so modules can be imported
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def calculate_integral(
    time, amplitude, energy_tag, apply_gain_correction=False
):

    if energy_tag == 20:
        if len(time) < 2 or len(amplitude) < 2:
            return float("nan")
        # sort by time to avoid negative/garbled integration when timestamps
        # are unsorted or wrapped
        order = np.argsort(time)
        time = np.asarray(time)[order]
        amplitude = np.asarray(amplitude)[order]
        baseline_window = min(50, len(amplitude))
        baseline = np.mean(amplitude[:baseline_window])
        # subtract baseline voltage
        amplitude = amplitude - baseline
        pulse_area = abs(np.trapezoid(amplitude, time))
        # correct for oscilloscope parameters
        amplifier_gain = 40  # db
        G = 10 ** (amplifier_gain / 20)
        charge = pulse_area / G / 50
        energy = charge * 7.83 / (1.602 * (10**-19)) / 1000  # keV

    else:
        pulse_area = abs(np.trapezoid(amplitude, time))
        if apply_gain_correction:
            amplifier_gain = 40  # db
            G = 10 ** (amplifier_gain / 20)
            charge = pulse_area / G / 50
            energy = charge * 7.83 / (1.602 * (10**-19)) / 1000  # keV
        else:
            energy = pulse_area * 7.83 / (1.602 * (10**-19)) / 1000

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


def load_bin_file(file_name):
    with open(file_name, "rb") as f:
        return f.read()


def strip_text_prefix_and_scpi(bin_data: bytes) -> bytes:
    hash_pos = bin_data.find(b"#")
    if hash_pos != -1:
        block = bin_data[hash_pos:]
        if len(block) < 3:
            raise ValueError("Invalid SCPI block header")

        n_digits_char = chr(block[1])
        if not n_digits_char.isdigit():
            raise ValueError(
                f"Invalid SCPI header digit count: {block[:16]!r}"
            )

        n_digits = int(n_digits_char)
        if len(block) < 2 + n_digits:
            raise ValueError("Incomplete SCPI block header")

        data_len = int(block[2 : 2 + n_digits].decode())
        start = 2 + n_digits
        end = start + data_len

        payload = block[start:end]
        if len(payload) != data_len:
            raise ValueError(
                f"SCPI block truncated, expected {data_len} bytes, got {len(payload)}"
            )

        return payload

    return bin_data


def split_segments(payload: bytes, nseg: int):
    if nseg <= 0:
        raise ValueError("nseg must be > 0")

    total_bytes = len(payload)
    if total_bytes % nseg != 0:
        raise ValueError(
            f"Payload lenth {total_bytes} is not divisible by NSEG={nseg}"
        )

    points_per_segment = total_bytes // nseg
    data = np.frombuffer(payload, dtype=np.int8).copy()
    data = data.reshape(nseg, points_per_segment)

    return data


def convert_to_volts(y_raw, gain, offset):
    return y_raw * gain - offset


def build_time_axis(npoints, dt=None, t0=0.0):
    if dt is None:
        return np.arange(npoints)
    return t0 + np.arange(npoints) * dt


def load_neutron_data(args, energy_tag: float):

    energies = []

    if not args.bias_voltage_v:
        return energies

    int_v = int(round(args.bias_voltage_v))

    if energy_tag == 20:

        for run_idx in range(3, 10):
            base_dir = (
                f"/home/claire/allpix-squared/NEUTRON_DATA/{int_v}V {run_idx}"
            )

            # iterate through each measurement file
            for meas_idx in range(1, 900):

                num = f"{meas_idx:05d}"

                file_path_1 = os.path.join(base_dir, f"C1meas{num}.txt")

                # vector of time values and amplitude values from the data files
                C1x, C1y = load_file_2col(file_path_1)

                # calculate area under pulse for bias voltage and convert to energy
                energy = calculate_integral(C1x, C1y, energy_tag)

                energies.append(energy)

    else:

        if energy_tag == 15.7:
            base_dir = "/home/claire/allpix-squared/NEUTRON_DATA/VDG/15.7 MeV/PN/PN/PN/segments"

        elif energy_tag == 17:
            base_dir = "/home/claire/allpix-squared/NEUTRON_DATA/VDG/17 MeV/PN/segments/segments"

        if not os.path.isdir(base_dir):
            print(
                f"WARNING: base_dir not found for {energy_tag} MeV: {base_dir}"
            )
            return energies

        for meas_idx in range(1, 500):
            num = f"{meas_idx:06d}"
            file_name = os.path.join(base_dir, f"seq_{num}.bin")
            nseg = 10
            try:
                bin_data = load_bin_file(file_name)
            except FileNotFoundError:
                continue
            payload = strip_text_prefix_and_scpi(bin_data)
            y_raw = split_segments(payload, nseg)
            nseg, npoints = y_raw.shape
            vertical_gain = 0.004
            vertical_offset = -0.02
            volts = convert_to_volts(y_raw, vertical_gain, vertical_offset)
            h_interval = 100e-12
            h_offset = 0.0
            x = build_time_axis(npoints, h_interval, h_offset)
            for trace in volts:
                energy = calculate_integral(
                    x, trace, energy_tag, apply_gain_correction=True
                )
                energies.append(energy)

    return energies


def main():

    # make function take arguemnts for bias voltage and distance from the command line

    parser = argparse.ArgumentParser(
        description="Scan bias voltage at fixed source distance."
    )
    parser.add_argument(
        "--bias-voltage-v",
        nargs="+",
        type=float,
        default=300,
        help="Bias voltage in V the experiment was conducted in.",
    )

    # distance of the source from the detector in the simulation
    parser.add_argument(
        "--distance-mm",
        type=float,
        default=30.00,
        help="Fixed source distance in mm for /gps/pos/centre z in source.mac.",
    )

    # path to the directory where csv files and plots are stored
    parser.add_argument(
        "--outdir",
        default="/home/claire/allpix-squared/Neutrons/ROOT_plots",
        help="Output directory for CSV and plots.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (defaults to <outdir>/experiment_heatmap.png).",
    )
    parser.add_argument(
        "--title",
        default="Heatmap: Source Energy vs Deposited Energy vs Counts (experiment)",
        help="Plot title.",
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
        default=15000,
        help="Maximum deposited energy (keV) to include (default: auto).",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=100.0,
        help="Deposited energy bin width in keV.",
    )

    args = parser.parse_args()

    # makes a directory to store the plots in
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # energy_tag is the experimental energy
    # REZ Experimental
    rez_energies = load_neutron_data(args, 20)

    # van der graff experiments
    vdg_15_energies = load_neutron_data(args, 15.7)

    vdg_17_energies = load_neutron_data(args, 17)

    items = [
        (20.0, rez_energies),
        (15.7, vdg_15_energies),
        (17.0, vdg_17_energies),
    ]
    for energy, samples in items:
        finite = int(np.sum(np.isfinite(samples)))
        print(
            f"Loaded {len(samples)} samples ({finite} finite) for {energy:g} MeV"
        )
        if finite > 0:
            samples_arr = np.asarray(samples, dtype=float)
            finite_vals = samples_arr[np.isfinite(samples_arr)]
            p1 = float(np.percentile(finite_vals, 1))
            p99 = float(np.percentile(finite_vals, 99))
            print(
                f"{energy:g} MeV deposited energy range (keV): "
                f"min={float(np.min(finite_vals)):.3g}, "
                f"p1={p1:.3g}, p99={p99:.3g}, "
                f"max={float(np.max(finite_vals)):.3g}"
            )

    xmin = args.xmin  # minimum deposited energy plotted
    bin_width = args.bin_width  # size of the grid
    if bin_width <= 0:
        raise RuntimeError("--bin-width must be > 0")

    if args.xmax is None:
        all_samples = np.concatenate(
            [np.asarray(s, dtype=float) for _, s in items]
        )
        finite_samples = all_samples[np.isfinite(all_samples)]
        if finite_samples.size == 0:
            raise RuntimeError("No finite deposited-energy samples found.")
        xmax = float(np.nanmax(finite_samples))
        # add a small headroom and snap to bin width
        xmax = np.ceil((xmax * 1.05) / bin_width) * bin_width
    else:
        xmax = args.xmax

    if xmax <= xmin:
        raise RuntimeError("--xmax must be greater than --xmin")
    edges = np.arange(xmin, xmax + bin_width, bin_width)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    # create a matrix of the source energies
    energies = [e for e, _ in items]
    energies_sorted = sorted(energies)
    energy_to_idx = {e: i for i, e in enumerate(energies_sorted)}

    # create a matrix of the counts per source energy
    counts_matrix = np.zeros(
        (len(energies_sorted), len(bin_centers)), dtype=float
    )

    for energy, samples in items:
        samples_arr = np.asarray(samples, dtype=float)
        in_range = np.sum(
            np.isfinite(samples_arr)
            & (samples_arr >= xmin)
            & (samples_arr < xmax)
        )
        print(
            f"{energy:g} MeV: {in_range} samples in range [{xmin}, {xmax}) keV"
        )
        for xc in samples:
            if not np.isfinite(xc):
                continue
            if xc < xmin or xc >= xmax:
                continue
            bin_index = int((xc - xmin) // bin_width)
            # add up all the counts from the source energy
            counts_matrix[energy_to_idx[energy], bin_index] += 1

    fig, ax = plt.subplots(figsize=(9, 6))
    z = counts_matrix.T
    z = np.where(z > 0, z, np.nan)

    # Use categorical x-bins so we always get one column per energy.
    energy_centers = np.asarray(energies_sorted, dtype=float)
    energy_edges = np.arange(energy_centers.size + 1, dtype=float)
    energy_tick_positions = energy_edges[:-1] + 0.5

    mesh = ax.pcolormesh(
        energy_edges,
        edges,
        z,
        shading="auto",
        norm=LogNorm(),
    )
    cbar = fig.colorbar(mesh, ax=ax)
    # use log scale for the counts
    cbar.set_label("Counts (log scale)")

    ax.set_xlabel("Source Energy (MeV)")
    ax.set_xticks(energy_tick_positions)
    ax.set_xticklabels([f"{e:g}" for e in energy_centers])
    ax.set_ylabel("Deposited Energy (keV)")
    ax.set_title(args.title)

    fig.tight_layout()
    out_path = (
        Path(args.out) if args.out else (outdir / "experiment_heatmap.png")
    )
    fig.savefig(out_path, dpi=160)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
