#!/usr/bin/env python3
"""
Compare deposited/collected "energy" spectra extracted from ROOT trees.

Trees compared (if present):
  - MCParticle: uses getTotalDepositedEnergy() (MeV) summed per event
  - DepositedCharge: converts charge carriers to energy using EHP
  - PropagatedCharge: same as DepositedCharge
  - PixelCharge: converts collected absolute charge to energy using EHP

Run command:
  conda run -n allpix2 python allpix-squared/SiC_3x3/compare_energy_trees.py <file.root>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_root_and_allpix(lib_path: str):
    import ROOT

    load_ret = ROOT.gSystem.Load(lib_path)
    # if load_ret < 0:
    #     raise RuntimeError(
    #         f"Failed to load Allpix objects library: {lib_path} (ret={load_ret})"
    #     )
    return ROOT


def pick_branch(tree, preferred: str) -> str:
    branches = [b.GetName() for b in tree.GetListOfBranches()]
    if preferred in branches:
        return preferred
    if f"{preferred}_0" in branches:
        return f"{preferred}_0"
    matches = [b for b in branches if b.startswith(preferred)]
    if len(matches) == 1:
        return matches[0]
    if len(branches) == 1:
        return branches[0]
    # go back to first branch
    return branches[0] if branches else ""


def event_energy_from_mcparticle(particles) -> float:
    total_mev = 0.0
    for p in particles:
        if p is None:
            continue
        if hasattr(p, "getTotalDepositedEnergy"):
            # sum the energy per event from deposited energy leaf
            total_mev += float(p.getTotalDepositedEnergy())
    return total_mev


def event_pairs_from_sensorcharge(charges) -> int:
    """
    Convert DepositedCharge/PropagatedCharge to number of e-h pairs.

    """
    electrons = 0
    holes = 0
    # sort into holes and electrons
    for c in charges:
        if c is None or not hasattr(c, "getCharge"):
            continue
        q = int(c.getCharge())
        sign = int(c.getSign()) if hasattr(c, "getSign") else 0
        if sign < 0:
            electrons += q
        elif sign > 0:
            holes += q
        else:
            # If no sign treat as pairs directly
            electrons += q

    if electrons > 0 and holes > 0:
        return min(electrons, holes)
    return electrons + holes


def event_abs_charge_from_pixelcharge(pixels) -> int:
    total = 0
    for p in pixels:
        if p is None:
            continue
        if hasattr(p, "getAbsoluteCharge"):
            # sum up all the charge per event
            total += int(p.getAbsoluteCharge())
        elif hasattr(p, "getCharge"):
            # add the charges from this leaf as well
            total += abs(int(p.getCharge()))
    return total


def extract_event_spectrum_keV(
    root,
    root_path: Path,
    tree_name: str,
    detector: str,
    ehp_energy_ev: float,
    max_entries: int | None,
    progress_every: int | None,
) -> np.ndarray:
    f = root.TFile(str(root_path), "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")

    t = f.Get(tree_name)
    if not t:
        raise RuntimeError(f"Tree '{tree_name}' not found in {root_path}")

    branch = pick_branch(t, detector)
    if not branch:
        raise RuntimeError(
            f"No usable branch found in tree '{tree_name}'. Available: "
            f"{[b.GetName() for b in t.GetListOfBranches()]}"
        )

    entries = int(t.GetEntries())
    if max_entries is not None:
        entries = min(entries, int(max_entries))

    out = []
    for i in range(entries):
        t.GetEntry(i)
        objs = getattr(t, branch)

        if tree_name == "MCParticle":
            mev = event_energy_from_mcparticle(objs)
            if mev > 0.0:
                out.append(mev * 1.0e3)  # converts to keV
        elif tree_name in ("DepositedCharge", "PropagatedCharge"):
            pairs = event_pairs_from_sensorcharge(objs)
            if pairs > 0:
                # convert charge to energy
                out.append(pairs * ehp_energy_ev * 1.0e-3)
        elif tree_name == "PixelCharge":
            q = event_abs_charge_from_pixelcharge(objs)
            if q > 0:
                # convert charge to energy
                out.append(q * ehp_energy_ev * 1.0e-3)
        else:
            raise RuntimeError(f"Unsupported tree: {tree_name}")

        if progress_every and (i + 1) % int(progress_every) == 0:
            print(f"  {tree_name}: processed {i + 1}/{entries} entries")

    return np.asarray(out, dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay deposited/collected energy spectra from multiple ROOT trees."
    )
    parser.add_argument(
        "root_file", help="Allpix Squared ROOT file to analyze."
    )
    parser.add_argument(
        "--detector",
        default="thedetector",
        help="Preferred detector branch name (falls back to first branch if not found).",
    )
    parser.add_argument(
        "--ehp-energy-ev",
        type=float,
        default=7.83,
        help="Energy per electron-hole pair in eV (SiC ~7.83 eV).",
    )
    parser.add_argument(
        "--lib",
        default="/home/claire/allpix-squared/lib/libAllpixObjects.so",
        help="Path to libAllpixObjects.so (dictionary for Allpix objects).",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries per tree to process (debug aid).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Print progress every N entries per tree.",
    )
    parser.add_argument(
        "--bins",
        type=float,
        default=50.0,
        help="Histogram bin width in keV (default: 200 keV).",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="Maximum x-axis in keV (default: auto).",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use log scale for y-axis.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: <root_file_stem>_tree_compare.png next to ROOT file).",
    )
    args = parser.parse_args()

    root_path = Path(args.root_file)
    if not root_path.exists():
        raise FileNotFoundError(root_path)

    root = _load_root_and_allpix(str(args.lib))

    trees = [
        "MCParticle",
        "DepositedCharge",
        "PropagatedCharge",
        "PixelCharge",
    ]
    spectra = {}
    for tname in trees:
        try:
            spectra[tname] = extract_event_spectrum_keV(
                root=root,
                root_path=root_path,
                tree_name=tname,
                detector=args.detector,
                ehp_energy_ev=float(args.ehp_energy_ev),
                max_entries=args.max_entries,
                progress_every=args.progress_every,
            )
            print(f"{tname}: events with signal = {spectra[tname].size}")
        except Exception as exc:
            print(f"{tname}: skipped ({exc})")

    if not spectra:
        raise RuntimeError("No spectra could be extracted.")

    # Determine plotting range:
    all_vals = np.concatenate([v for v in spectra.values() if v.size > 0])
    if all_vals.size == 0:
        raise RuntimeError("All extracted spectra are empty.")

    xmin = 0.0
    xmax = (
        float(args.xmax)
        if args.xmax is not None
        else float(np.nanmax(all_vals) * 1.05)
    )
    if not (xmax > xmin):
        xmax = 1.0

    bin_w = float(args.bins)
    edges = np.arange(xmin, xmax + bin_w, bin_w)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 6))
    for name in trees:
        data = spectra.get(name)
        if data is None or data.size == 0:
            continue
        plt.hist(
            data,
            bins=edges,
            histtype="step",
            linewidth=1.3,
            label=name,
        )

    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.title(f"Tree comparison: {root_path.name}")
    plt.legend()
    if args.logy:
        plt.yscale("log")
    plt.tight_layout()

    out_path = (
        Path(args.out)
        if args.out
        else root_path.with_name(f"{root_path.stem}_tree_compare.png")
    )
    plt.savefig(out_path, dpi=160)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
