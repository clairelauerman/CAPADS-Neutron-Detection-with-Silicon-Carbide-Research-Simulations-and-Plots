# -*- coding: utf-8 -*-
import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =====================
# USER SETTINGS
# =====================
NSEG = 50  # počet segmentů v dumpu
PLOT_MODE = "raw"  # "raw" nebo "volts"
MAX_PLOTS = 1  # kolik jednotlivých segmentů vykreslit
OVERLAY_COUNT = 100  # kolik segmentů dát do overlay grafu

# Volitelné škálování do voltů, pokud ho znáš
VERTICAL_GAIN = None  # např. 0.004
VERTICAL_OFFSET = None  # např. -0.02

# Volitelná časová osa
HORIZ_INTERVAL = None  # např. 100e-12
HORIZ_OFFSET = 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--energy",
        type=float,
        default=16.0,
        help="Neutron source energy in MeV (e.g., 15.7, 17, 20).",
    )
    parser.add_argument(
        "-n",
        "--n_seg",
        type=int,
        default=None,
        help="Plot segment -n",
    )
    parser.add_argument(
        "--input-dir",
        default="/home/claire/allpix-squared/NEUTRON_DATA/VDG/16 MeV/data1/segments",
        help="Directory with seq_*.bin files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1000,
        help="Maximum number of files to process.",
    )
    parser.add_argument(
        "--sim-csv",
        default="/home/claire/allpix-squared/Neutrons/ROOT_plots/h_deposited_energy_16.0MeV.csv",
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
        "--bin-width",
        type=float,
        default=None,
        help="Histogram bin width in keV (default: inferred from sim CSV).",
    )
    parser.set_defaults(logy=True)
    parser.add_argument(
        "--logy",
        dest="logy",
        action="store_true",
        help="Use log scale on the simulation y-axis (default).",
    )

    args = parser.parse_args()

    if args.n_seg is not None and args.n_seg <= 0:
        parser.error("-n must be a positive integer")

    return args


# reads the simulation csvs
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


def load_raw_file(filename):
    with open(filename, "rb") as f:
        return f.read()


def strip_text_prefix_and_scpi(raw: bytes) -> bytes:
    """
    Supports files like:
        DAT1,#9000402000<binary data>
    or directly:
        #9000402000<binary data>
    or just raw binary data.

    Returns binary payload only.
    """
    # 1) pokud je tam textový prefix a někde dál začíná SCPI blok '#'
    hash_pos = raw.find(b"#")
    if hash_pos != -1:
        block = raw[hash_pos:]
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

    # 2) fallback, žádný SCPI header nenalezen, ber raw jako payload
    return raw


def extract_scpi_blocks(raw: bytes):
    blocks = []
    i = 0

    while i < len(raw):
        if raw[i : i + 1] != b"#":
            i += 1
            continue

        n_digits = int(chr(raw[i + 1]))
        data_len = int(raw[i + 2 : i + 2 + n_digits].decode())

        start = i + 2 + n_digits
        end = start + data_len

        blocks.append(raw[i:end])
        i = end

    return blocks


def split_two_channels(payload: bytes):
    half = len(payload) // 2
    c1 = payload[:half]
    c2 = payload[half:]
    return c1, c2


def split_segments(payload: bytes, nseg: int):
    if nseg <= 0:
        raise ValueError("nseg must be > 0")

    total_bytes = len(payload)
    if total_bytes % nseg != 0:
        raise ValueError(
            f"Payload length {total_bytes} is not divisible by NSEG={nseg}"
        )

    points_per_segment = total_bytes // nseg

    # BYTE format => int8
    data = np.frombuffer(payload, dtype=np.int8).copy()
    data = data.reshape(nseg, points_per_segment)

    return data


def convert_to_volts(y_raw, gain, offset):
    if gain is None or offset is None:
        raise ValueError(
            "VERTICAL_GAIN and VERTICAL_OFFSET must be set for PLOT_MODE='volts'"
        )
    return y_raw * gain - offset


def build_time_axis(npoints, dt=None, t0=0.0):
    if dt is None:
        return np.arange(npoints)
    return t0 + np.arange(npoints) * dt


def subtract_baseline(amps):
    result = []
    for amp in amps:
        # convert the values into a 32 bit integer
        # amp = amp.astype(np.int32)
        # average the first 15 values in the sample together
        baseline = np.mean(amp[:15])

        # subtract that average from the values
        result.append(amp - baseline)
    return np.array(result)


def filter_triggered(amps, threshold):
    trigg_amps = []
    for amp in amps:
        max_value = max(amp)
        if max_value > threshold:
            trigg_amps.append(amp)
        else:
            continue
    return np.array(trigg_amps)


def find_peak_boundaries(amp):
    peak_index = np.argmax(amp)
    peak_low_level = 0.05 * max(amp)
    # print(max(amp))

    # LEFT
    index_left = peak_index
    while index_left > 0 and amp[index_left - 1] > peak_low_level:
        index_left -= 1

    # RIGHT
    index_right = peak_index
    length = len(amp)

    while index_right < length - 1 and amp[index_right + 1] > peak_low_level:
        index_right += 1
    return index_left, index_right


def get_spectrum(amps, time):
    spectrum = []
    # convert time from nanoseconds into seconds
    time_sec = time * 1e-9
    for amp in amps:
        left, right = find_peak_boundaries(amp)
        # numerical integration of the waveform
        area = abs(
            np.trapezoid(amp[left : right + 1], time_sec[left : right + 1])
        )
        amplifier_gain = 40  # db
        # correct for amplifier gain
        G = 10 ** (amplifier_gain / 20)
        # convert pulse area into charge
        charge = area / G / 50
        # convert charge into energy
        energy = charge * 7.83 / (1.602 * (10**-19)) / 1e6
        spectrum.append(energy)
    spectrum = np.array(spectrum)
    return spectrum


def plot_histogram(
    hist_val,
    begin_val,
    end_val,
    step_val,
    plt_scale,
    title,
    xaxis,
    yaxis,
    label1,
):
    hist_val = np.asarray(hist_val, dtype=float)
    hist_val = hist_val[np.isfinite(hist_val)]
    if hist_val.size == 0:
        raise RuntimeError(
            "Histogram input is empty after filtering invalid values."
        )

    plt.hist(
        hist_val,
        bins=np.arange(begin_val, end_val, step_val),
        histtype="step",
        alpha=0.7,
        label=label1,
    )

    plt.xlabel(xaxis, fontsize=16)
    plt.ylabel(yaxis, fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.legend(fontsize=16)
    plt.title(title, fontsize=16)
    if plt_scale is True:
        plt.yscale("log")
    plt.show()


def get_time_amps(
    filepath, howmany
):  # get waveforms' values from all seq_, time x is same everywhere
    result_c1 = []
    result_c2 = []

    x = None

    folder = filepath
    # Find all matching files
    files = glob.glob(os.path.join(folder, "seq_*.bin"))
    if len(files) == 0:
        raise RuntimeError(f"No seq_*.bin files found in: {folder}")

    # Sort numerically by sequence number
    files.sort(key=lambda x: int(os.path.basename(x)[4:10]))
    i = 0
    for FILENAME in files:
        if i >= howmany:
            break
        raw = load_raw_file(FILENAME)

        # payload = strip_text_prefix_and_scpi(raw)

        blocks = extract_scpi_blocks(raw)

        c1_raw = split_segments(strip_text_prefix_and_scpi(blocks[0]), NSEG)

        c2_raw = split_segments(strip_text_prefix_and_scpi(blocks[1]), NSEG)

        nseg, npoints = c1_raw.shape

        if PLOT_MODE == "volts":
            c1 = convert_to_volts(c1_raw, VERTICAL_GAIN, VERTICAL_OFFSET)
            c2 = convert_to_volts(c2_raw, VERTICAL_GAIN, VERTICAL_OFFSET)
            ylabel = "Voltage [V]"
        else:
            c1 = c1_raw
            c2 = c2_raw
            ylabel = "ADC code / raw byte"

        x = build_time_axis(npoints, HORIZ_INTERVAL, HORIZ_OFFSET)

        result_c1.append(c1)
        result_c2.append(c2)
        i += 1
    x = np.array(x)
    result_c1 = np.array(result_c1)
    result_c2 = np.array(result_c2)

    return x, result_c1, result_c2


def get_spectrum_files(filepath, howmany):
    x, c1_all, c2_all = get_time_amps(filepath, howmany)
    if len(c1_all) == 0 or len(c2_all) == 0:
        raise RuntimeError(f"No waveform segments loaded from: {filepath}")
    overall_spectrum_c1 = []
    overall_spectrum_c2 = []

    # len(c1_all)
    for dir_dat in range(len(c1_all)):

        c1 = c1_all[dir_dat]
        c2 = c2_all[dir_dat]

        c1 = subtract_baseline(c1)
        c2 = subtract_baseline(c2)

        c1 = filter_triggered(c1, 10)
        c2 = filter_triggered(c2, 10)

        spect_c1 = get_spectrum(c1, x)
        spect_c2 = get_spectrum(c2, x)

        spect_c1 = np.array(spect_c1)
        spect_c2 = np.array(spect_c2)

        if spect_c1.size > 0:
            overall_spectrum_c1.append(spect_c1)
        if spect_c2.size > 0:
            overall_spectrum_c2.append(spect_c2)
    if len(overall_spectrum_c1) == 0 or len(overall_spectrum_c2) == 0:
        raise RuntimeError(
            "No triggered pulses found after filtering. "
            "Try lowering the trigger threshold in filter_triggered()."
        )
    overall_spectrum_c1 = np.concatenate(overall_spectrum_c1)
    overall_spectrum_c2 = np.concatenate(overall_spectrum_c2)

    return overall_spectrum_c1, overall_spectrum_c2


def main():
    args = parse_args()
    spect1, spect2 = get_spectrum_files(args.input_dir, args.max_files)

    print("len:", len(spect1))
    print("min/max:", np.min(spect1), np.max(spect1))

    plot_histogram(
        spect1,
        0,
        14000,
        10,
        True,
        "16.0 MeV Neutron Detection by 4H-SiC PN Detector",
        "Deposited Energy (keV)",
        "Counts",
        "PN",
    )
    # plot_histogram(
    #     spect2,
    #     0,
    #     170,
    #     1,
    #     True,
    #     "title",
    #     "ADC code / raw byte",
    #     "Counts",
    #     "LGAD",
    # )

    sim_csv = Path(args.sim_csv)
    if not sim_csv.exists():
        raise RuntimeError(f"Simulation CSV not found: {sim_csv}")

    # Load simulation histogram (already binned)
    sim_x, sim_counts = read_hist_csv(sim_csv)
    sim_x = np.asarray(sim_x, dtype=float)
    sim_counts = np.asarray(sim_counts, dtype=float)

    spect1 = np.asarray(spect1, dtype=float)
    spect1 = spect1[np.isfinite(spect1)]
    if spect1.size == 0:
        raise RuntimeError(
            "Histogram input is empty after filtering invalid values."
        )

    bins = np.arange(0, 14000, 10)

    # simulation: x = bin centers, weights = counts in each bin
    plt.hist(
        sim_x,
        bins=bins,
        weights=sim_counts,
        histtype="step",
        alpha=0.7,
        label="Simulation",
    )

    # experiment
    plt.hist(
        spect1,
        bins=bins,
        histtype="step",
        alpha=0.7,
        label="Experiment",
    )

    plt.xlabel("Energy (keV)", fontsize=16)
    plt.ylabel("Counts", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.legend(fontsize=16)
    plt.title("16.0 MeV Neutron Simulation vs Experiment (-350V)", fontsize=16)
    plt.yscale("log")
    plt.show()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else outdir / f"overlay_{args.energy:g}MeV.png"
    )
    plt.savefig(out_path, dpi=160)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
