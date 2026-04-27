#!/usr/bin/env python3
import argparse
import os
import os.path as path
import re

import ROOT
from ROOT import gSystem


def find_allpix_objects_lib():
    for p in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        lib = path.join(p, "libAllpixObjects.so")
        if path.isfile(lib):
            return lib
    return None


def pick_branch(tree, detector):
    branches = [b.GetName() for b in tree.GetListOfBranches()]
    if detector in branches:
        return detector
    # Common pattern: <detector>_0
    if f"{detector}_0" in branches:
        return f"{detector}_0"
    # Fallback: pick the only branch that starts with detector
    matches = [b for b in branches if b.startswith(detector)]
    if len(matches) == 1:
        return matches[0]
    if len(branches) == 1:
        return branches[0]
    return None


def count_particles(root_path, detector, tree_name, particle_id):
    f = ROOT.TFile(root_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")

    tree = f.Get(tree_name)
    if not tree:
        raise RuntimeError(f"Tree '{tree_name}' not found in {root_path}")

    branch_name = pick_branch(tree, detector)
    if not branch_name:
        available = [b.GetName() for b in tree.GetListOfBranches()]
        raise RuntimeError(
            f"Could not find detector branch for '{detector}'. "
            f"Available branches: {available}"
        )

    total = 0
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        br = getattr(tree, branch_name)
        if particle_id is None:
            total += len(br)
            continue
        # Filter only if objects provide getParticleID (MCParticle)
        for obj in br:
            if (
                hasattr(obj, "getParticleID")
                and obj.getParticleID() == particle_id
            ):
                total += 1
    return total, branch_name, tree.GetEntries()


def parse_energy_from_filename(root_path):
    base = path.basename(root_path)
    match = re.search(r"_([0-9]+(?:\\.[0-9]+)?)MeV", base)
    if not match:
        return None
    return float(match.group(1))


def deposited_energy_histogram(
    root_path,
    detector,
    tree_name,
    particle_id,
    dep_min,
    dep_max,
    dep_bins,
    max_entries,
    progress_every,
    dep_threshold,
    ehp_energy_ev,
):
    f = ROOT.TFile(root_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")

    tree = f.Get(tree_name)
    if not tree:
        raise RuntimeError(f"Tree '{tree_name}' not found in {root_path}")

    branch_name = pick_branch(tree, detector)
    if not branch_name:
        available = [b.GetName() for b in tree.GetListOfBranches()]
        raise RuntimeError(
            f"Could not find detector branch for '{detector}'. "
            f"Available branches: {available}"
        )

    if dep_max <= dep_min:
        raise RuntimeError("--dep-max must be greater than --dep-min")

    width = (dep_max - dep_min) / dep_bins
    counts = [0] * dep_bins
    total = 0

    entries = tree.GetEntries()
    if max_entries is not None:
        entries = min(entries, max_entries)

    for i in range(entries):
        tree.GetEntry(i)
        br = getattr(tree, branch_name)
        for obj in br:
            if particle_id is not None:
                if not hasattr(obj, "getParticleID"):
                    continue
                if obj.getParticleID() != particle_id:
                    continue
            e = get_deposited_energy_mev(obj, tree_name, ehp_energy_ev)
            if e <= dep_threshold:
                continue
            if e < dep_min or e > dep_max:
                continue
            bin_index = int((e - dep_min) / width)
            if bin_index >= dep_bins:
                bin_index = dep_bins - 1
            counts[bin_index] += 1
            total += 1
        if progress_every is not None and progress_every > 0:
            if (i + 1) % progress_every == 0:
                print(f"  processed {i + 1}/{entries} entries")

    edges = [dep_min + i * width for i in range(dep_bins + 1)]
    return counts, edges, branch_name, entries, total


def deposited_energy_sum(
    root_path,
    detector,
    tree_name,
    particle_id,
    dep_threshold,
    max_entries,
    progress_every,
    ehp_energy_ev,
):
    f = ROOT.TFile(root_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")

    tree = f.Get(tree_name)
    if not tree:
        raise RuntimeError(f"Tree '{tree_name}' not found in {root_path}")

    branch_name = pick_branch(tree, detector)
    if not branch_name:
        available = [b.GetName() for b in tree.GetListOfBranches()]
        raise RuntimeError(
            f"Could not find detector branch for '{detector}'. "
            f"Available branches: {available}"
        )

    entries = tree.GetEntries()
    if max_entries is not None:
        entries = min(entries, max_entries)

    total_deposited = 0.0
    total_particles = 0

    for i in range(entries):
        tree.GetEntry(i)
        br = getattr(tree, branch_name)
        for obj in br:
            if particle_id is not None:
                if not hasattr(obj, "getParticleID"):
                    continue
                if obj.getParticleID() != particle_id:
                    continue
            e = get_deposited_energy_mev(obj, tree_name, ehp_energy_ev)
            if e <= dep_threshold:
                continue
            total_deposited += e
            total_particles += 1
        if progress_every is not None and progress_every > 0:
            if (i + 1) % progress_every == 0:
                print(f"  processed {i + 1}/{entries} entries")

    return total_deposited, total_particles, branch_name, entries


def get_deposited_energy_mev(obj, tree_name, ehp_energy_ev):
    if hasattr(obj, "getTotalDepositedEnergy"):
        return obj.getTotalDepositedEnergy()
    # PixelCharge case: convert charge (e) to energy using e-h pair energy
    if hasattr(obj, "getAbsoluteCharge") or hasattr(obj, "getCharge"):
        if ehp_energy_ev is None:
            raise RuntimeError(
                f"Tree '{tree_name}' provides charge but not deposited energy. "
                "Provide --ehp-energy-ev to convert charge to energy."
            )
        if hasattr(obj, "getAbsoluteCharge"):
            charge = obj.getAbsoluteCharge()
        else:
            charge = abs(obj.getCharge())
        return charge * ehp_energy_ev * 1e-6
    raise RuntimeError(
        f"Objects in tree '{tree_name}' do not provide deposited energy or charge"
    )


def pdg_and_process_breakdown(
    root_path,
    detector,
    tree_name,
    particle_id_filter=None,
    max_entries=None,
    progress_every=None,
    include_process=False,
    exclude_process_none=False,
):
    f = ROOT.TFile(root_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")

    tree = f.Get(tree_name)
    if not tree:
        raise RuntimeError(f"Tree '{tree_name}' not found in {root_path}")

    branch_name = pick_branch(tree, detector)
    if not branch_name:
        available = [b.GetName() for b in tree.GetListOfBranches()]
        raise RuntimeError(
            f"Could not find detector branch for '{detector}'. "
            f"Available branches: {available}"
        )

    entries = tree.GetEntries()
    if max_entries is not None:
        entries = min(entries, max_entries)

    pdg_counts = {}
    process_counts = {}
    total_objects = 0

    for i in range(entries):
        tree.GetEntry(i)
        br = getattr(tree, branch_name)
        for obj in br:
            if not hasattr(obj, "getParticleID"):
                continue
            pid = int(obj.getParticleID())
            if particle_id_filter is not None and pid != int(particle_id_filter):
                continue
            pdg_counts[pid] = pdg_counts.get(pid, 0) + 1
            total_objects += 1

            if include_process:
                # MCTrack objects store the creator process directly:
                if hasattr(obj, "getCreationProcessName"):
                    proc = str(obj.getCreationProcessName())
                # MCParticle stores a pointer to MCTrack (may be null if history not available):
                elif hasattr(obj, "getTrack"):
                    tr = obj.getTrack()
                    if tr and hasattr(tr, "getCreationProcessName"):
                        proc = str(tr.getCreationProcessName())
                    else:
                        proc = "none"
                else:
                    proc = "none"
                if exclude_process_none and proc == "none":
                    continue
                process_counts[proc] = process_counts.get(proc, 0) + 1

        if progress_every is not None and progress_every > 0:
            if (i + 1) % progress_every == 0:
                print(f"  processed {i + 1}/{entries} entries")

    return pdg_counts, process_counts, branch_name, entries, total_objects


def main():
    parser = argparse.ArgumentParser(
        description="Count particles hitting a detector from an Allpix Squared ROOT file"
    )
    parser.add_argument(
        "root_files",
        nargs="+",
        help="Path(s) to ROOT file(s), e.g. output/Neutrons_30mm_26MeV.root",
    )
    parser.add_argument(
        "--detector",
        default="neutron_detector",
        help="Detector name from the geometry config (default: neutron_detector)",
    )
    parser.add_argument(
        "--tree",
        default="MCParticle",
        help="Tree name to count from (default: MCParticle). "
        "Other options: PixelHit, DepositedCharge, PropagatedCharge",
    )
    parser.add_argument(
        "--particle-id",
        type=int,
        default=2112,
        help="PDG particle ID to count (default: 2112 for neutrons). "
        "Set to -1 to count all particles in the branch.",
    )
    parser.add_argument(
        "--hist-out",
        default=None,
        help="Optional output path for a histogram PNG (requires matplotlib). "
        "Histogram bins are per file energy parsed from filename (hist-mode=source) "
        "or deposited energy (hist-mode=deposited).",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV output (energy_MeV,count).",
    )
    parser.add_argument(
        "--hist-mode",
        choices=["source", "deposited", "source-deposited"],
        default="source",
        help="Histogram mode: source = counts per source energy; "
        "deposited = deposited energy spectrum per source energy; "
        "source-deposited = total deposited energy per source energy "
        "(default: source).",
    )
    parser.add_argument(
        "--dep-bins",
        type=int,
        default=50,
        help="Number of bins for deposited-energy histogram (default: 50).",
    )
    parser.add_argument(
        "--dep-min",
        type=float,
        default=None,
        help="Minimum deposited energy (MeV). Default: 0.",
    )
    parser.add_argument(
        "--dep-max",
        type=float,
        default=None,
        help="Maximum deposited energy (MeV). Default: source energy parsed from filename.",
    )
    parser.add_argument(
        "--dep-threshold",
        type=float,
        default=0.0,
        help="Minimum deposited energy (MeV) to include (default: 0).",
    )
    parser.add_argument(
        "--ehp-energy-ev",
        type=float,
        default=None,
        help="Energy per electron-hole pair (eV). Required when using PixelCharge "
        "to convert charge to deposited energy.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of tree entries to process per file (debug aid).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Print progress every N entries (debug aid).",
    )
    parser.add_argument(
        "--min-energy",
        type=float,
        default=None,
        help="Minimum energy (MeV) parsed from filename to include.",
    )
    parser.add_argument(
        "--max-energy",
        type=float,
        default=None,
        help="Maximum energy (MeV) parsed from filename to include.",
    )
    parser.add_argument(
        "--unique-energy",
        action="store_true",
        help="If multiple files share the same energy, only the first is processed.",
    )
    parser.add_argument(
        "--pdg-breakdown",
        action="store_true",
        help="Print a PDG-code breakdown for objects in the selected tree/branch "
        "(requires objects with getParticleID(), e.g. MCParticle/MCTrack).",
    )
    parser.add_argument(
        "--process-breakdown",
        action="store_true",
        help="Also print a Geant4 creator-process breakdown "
        "(MCParticle only; uses getTrack()->getCreationProcessName()).",
    )
    parser.add_argument(
        "--exclude-process-none",
        action="store_true",
        help="Exclude creator process 'none' from process breakdown/histograms (useful to drop primaries).",
    )
    parser.add_argument(
        "--process-hist-out",
        default=None,
        help="Optional output PNG path for a creator-process histogram (bar chart). "
        "Aggregates across all processed input files.",
    )
    parser.add_argument(
        "--process-csv-out",
        default=None,
        help="Optional output CSV path for creator-process counts. "
        "Aggregates across all processed input files.",
    )
    parser.add_argument(
        "--process-top",
        type=int,
        default=30,
        help="How many top processes to print/plot (default: 30).",
    )
    parser.add_argument(
        "--lib",
        default=None,
        help="Path to libAllpixObjects.so (optional if in LD_LIBRARY_PATH)",
    )
    args = parser.parse_args()

    lib = args.lib or find_allpix_objects_lib()
    if not lib or not path.isfile(lib):
        raise RuntimeError(
            "libAllpixObjects.so not found. "
            "Pass --lib /path/to/libAllpixObjects.so or add it to LD_LIBRARY_PATH."
        )

    gSystem.Load(lib)

    particle_id = None if args.particle_id == -1 else args.particle_id

    # De-duplicate input files while preserving order
    ordered_files = list(dict.fromkeys(args.root_files))
    # Prefer sorting by parsed energy when available for deterministic order
    ordered_files.sort(
        key=lambda p: (
            parse_energy_from_filename(p) is None,
            parse_energy_from_filename(p),
        )
    )

    print(f"Processing {len(ordered_files)} files")

    results = []
    seen_energies = set()
    aggregated_process_counts = {}
    for root_path in ordered_files:
        energy = parse_energy_from_filename(root_path)
        if energy is None and (
            args.min_energy is not None or args.max_energy is not None
        ):
            continue
        if (
            args.min_energy is not None
            and energy is not None
            and energy < args.min_energy
        ):
            continue
        if (
            args.max_energy is not None
            and energy is not None
            and energy > args.max_energy
        ):
            continue
        if args.unique_energy and energy is not None:
            if energy in seen_energies:
                continue
            seen_energies.add(energy)

        if args.hist_mode == "deposited":
            dep_min = 0.0 if args.dep_min is None else args.dep_min
            dep_max = args.dep_max
            if dep_max is None:
                if energy is None:
                    raise RuntimeError(
                        "Could not infer --dep-max from filename. "
                        "Provide --dep-max explicitly."
                    )
                dep_max = energy
            counts, edges, branch_name, entries, total = (
                deposited_energy_histogram(
                    root_path,
                    args.detector,
                    args.tree,
                    particle_id,
                    dep_min,
                    dep_max,
                    args.dep_bins,
                    args.max_entries,
                    args.progress_every,
                    args.dep_threshold,
                    args.ehp_energy_ev,
                )
            )
            results.append(
                (
                    root_path,
                    energy,
                    total,
                    branch_name,
                    entries,
                    counts,
                    edges,
                    None,
                )
            )
            mode_total = total
        elif args.hist_mode == "source-deposited":
            total_dep, total_particles, branch_name, entries = (
                deposited_energy_sum(
                    root_path,
                    args.detector,
                    args.tree,
                    particle_id,
                    args.dep_threshold,
                    args.max_entries,
                    args.progress_every,
                    args.ehp_energy_ev,
                )
            )
            results.append(
                (
                    root_path,
                    energy,
                    total_dep,
                    branch_name,
                    entries,
                    total_particles,
                    None,
                    None,
                )
            )
            mode_total = total_particles
        else:
            total, branch_name, entries = count_particles(
                root_path, args.detector, args.tree, particle_id
            )
            results.append(
                (
                    root_path,
                    energy,
                    total,
                    branch_name,
                    entries,
                    None,
                    None,
                    None,
                )
            )
            mode_total = total

        print(f"File: {root_path}")
        print(f"Tree: {args.tree}")
        print(f"Branch: {branch_name}")
        print(f"Entries: {entries}")
        if particle_id is None:
            print("Particle filter: none")
        else:
            print(f"Particle filter (PDG): {particle_id}")
        if args.hist_mode == "source-deposited":
            total_deposited = results[-1][2]
            print(f"Total deposited energy (MeV): {total_deposited}")
            print(f"Total particles counted: {mode_total}")
        else:
            print(f"Total particles counted: {mode_total}")

        wants_process = (
            args.process_breakdown
            or args.process_hist_out is not None
            or args.process_csv_out is not None
        )
        if args.pdg_breakdown or wants_process:
            pdg_counts, proc_counts, _, _, total_objs = pdg_and_process_breakdown(
                root_path,
                args.detector,
                args.tree,
                particle_id_filter=particle_id,
                max_entries=args.max_entries,
                progress_every=args.progress_every,
                include_process=bool(wants_process),
                exclude_process_none=bool(args.exclude_process_none),
            )
            if args.pdg_breakdown:
                print(f"PDG objects tallied: {total_objs}")
                for pid, cnt in sorted(
                    pdg_counts.items(), key=lambda kv: kv[1], reverse=True
                )[:20]:
                    print(f"  PDG {pid}: {cnt}")
            if args.process_breakdown:
                for proc, cnt in sorted(
                    proc_counts.items(), key=lambda kv: kv[1], reverse=True
                )[: args.process_top]:
                    print(f"  Process {proc}: {cnt}")
            for proc, cnt in proc_counts.items():
                aggregated_process_counts[proc] = (
                    aggregated_process_counts.get(proc, 0) + cnt
                )
        print("")

    if (args.process_hist_out or args.process_csv_out) and not aggregated_process_counts:
        raise RuntimeError(
            "No process information collected. "
            "Try --tree MCTrack (process stored directly) or --tree MCParticle with history enabled."
        )

    if args.process_csv_out:
        with open(args.process_csv_out, "w", encoding="utf-8") as f:
            f.write("process,count\n")
            for proc, cnt in sorted(
                aggregated_process_counts.items(),
                key=lambda kv: kv[1],
                reverse=True,
            ):
                f.write(f"{proc},{cnt}\n")

    if args.process_hist_out:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                "matplotlib is required for --process-hist-out"
            ) from exc

        top = sorted(
            aggregated_process_counts.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[: max(1, int(args.process_top))]
        labels = [p for p, _ in top]
        values = [c for _, c in top]

        plt.figure(figsize=(10, max(3.0, 0.28 * len(labels))))
        y = range(len(labels))
        plt.barh(y, values)
        plt.yticks(y, labels)
        plt.gca().invert_yaxis()
        plt.xlabel("Count")
        plt.title(
            f"Creator-process counts in {args.tree} ({args.detector})"
            + (" (excluding 'none')" if args.exclude_process_none else "")
        )
        plt.tight_layout()
        plt.savefig(args.process_hist_out, dpi=160)

    if args.csv_out:
        with open(args.csv_out, "w", encoding="utf-8") as f:
            if args.hist_mode == "source-deposited":
                f.write("energy_MeV,total_deposited_MeV,root_file\n")
            else:
                f.write("energy_MeV,count,root_file\n")
            for root_path, energy, total, _, _, _, _, _ in results:
                energy_str = "" if energy is None else str(energy)
                f.write(f"{energy_str},{total},{root_path}\n")

    if args.hist_out:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                "matplotlib is required for --hist-out"
            ) from exc

        if args.hist_mode == "deposited":
            plot_data = [
                (e, cts, edges)
                for _, e, _, _, _, cts, edges, _ in results
                if cts is not None and edges is not None
            ]
            if not plot_data:
                raise RuntimeError(
                    "No deposited-energy data available to plot."
                )
            plot_data.sort(key=lambda x: (x[0] is None, x[0]))

            plt.figure(figsize=(8, 5))
            for energy, counts, edges in plot_data:
                centers = [
                    (edges[i] + edges[i + 1]) / 2 for i in range(len(counts))
                ]
                label = (
                    f"{energy} MeV" if energy is not None else "Unknown MeV"
                )
                plt.step(centers, counts, where="mid", label=label)

            plt.xlabel("Deposited Energy (MeV)")
            plt.ylabel("Counts")
            title = f"{args.tree} deposited-energy spectrum in {args.detector}"
            if particle_id is not None:
                title += f" (PDG {particle_id})"
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.hist_out, dpi=160)
        elif args.hist_mode == "source-deposited":
            plot_data = [
                (e, c) for _, e, c, _, _, _, _, _ in results if e is not None
            ]
            if not plot_data:
                raise RuntimeError(
                    "No energies parsed from filenames. Expected pattern like *_26MeV.root"
                )
            plot_data.sort(key=lambda x: x[0])
            energies = [e for e, _ in plot_data]
            totals = [c for _, c in plot_data]

            plt.figure(figsize=(7, 4))
            plt.bar(energies, totals, width=0.8)
            plt.xlabel("Neutron Energy (MeV)")
            plt.ylabel("Total Deposited Energy (MeV)")
            title = f"{args.tree} total deposited energy in {args.detector}"
            if particle_id is not None:
                title += f" (PDG {particle_id})"
            if args.dep_threshold > 0:
                title += f", threshold>{args.dep_threshold:g} MeV"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(args.hist_out, dpi=160)
        else:
            # Only plot files where energy could be parsed
            plot_data = [
                (e, c) for _, e, c, _, _, _, _, _ in results if e is not None
            ]
            if not plot_data:
                raise RuntimeError(
                    "No energies parsed from filenames. Expected pattern like *_26MeV.root"
                )
            plot_data.sort(key=lambda x: x[0])
            energies = [e for e, _ in plot_data]
            counts = [c for _, c in plot_data]

            plt.figure(figsize=(7, 4))
            plt.bar(energies, counts, width=0.8)
            plt.xlabel("Neutron Energy (MeV)")
            plt.ylabel("Counts")
            title = f"{args.tree} counts in {args.detector}"
            if particle_id is not None:
                title += f" (PDG {particle_id})"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(args.hist_out, dpi=160)


if __name__ == "__main__":
    main()
