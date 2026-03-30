# Runs the allpix simulation to generate new ROOT files for different geometries
# depending on the inputs
import argparse
import re
import subprocess
import sys
from pathlib import Path

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
        f"/gps/pos/centre {distance_mm:g} 0 0 mm",
        original,
        flags=re.MULTILINE,
    )
    if count_center == 0:
        updated = original.replace(
            "/gps/pos/type Point",
            f"/gps/pos/type Point\n/gps/pos/centre {distance_mm:g} 0 0 mm",
        )
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


def get_electric_field(v: float) -> str:
    v_int = int(round(v))
    return (
        f"/home/claire/allpix-squared/Neutrons/Meshes/"
        f"PN_bias_{v_int:04d}_des_ElectricField.apf"
    )


# set the source positon to 0 in the main config bc it will be overwritten
# by the source.mac file
def set_conf_bias_and_output(
    original_conf: str,
    bias_v: float,
    distance_mm: float,
    material: str,
    electric_field: str,
) -> str:
    updated = re.sub(
        r"^\s*source_position\s*=\s*.*\{DISTANCE\}.*$",
        "source_position = 0um 0um 0um",
        original_conf,
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

    # Set correct electric field TCAD file in the configuration
    def replace_field_block(match: re.Match) -> str:
        block = match.group(0)
        block, replaced = re.subn(
            r"^\s*file_name\s*=.*$",
            f'file_name = "{electric_field}"',
            block,
            flags=re.MULTILINE,
        )
        if replaced == 0:
            block = block.rstrip() + f'\nfile_name = "{electric_field}"\n'
        return block

    updated, count_field = re.subn(
        r"(?ms)^\[ElectricFieldReader\].*?(?=^\[|\Z)",
        replace_field_block,
        updated,
    )
    if count_field == 0:
        raise RuntimeError(
            "Could not update ElectricFieldReader file_name in config"
        )

    # make distance and bias voltage integers
    distance_int = int(round(distance_mm))
    bias_int = int(round(bias_v))

    # create the correct name of root file to be written
    replacement = f'file_name = "Am241alpha_{material}_{distance_int}mm_{bias_int}V.root"'

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


def run_bias_scan(
    args,
    conf_path,
    source_mac_path,
):

    original_conf = conf_path.read_text()
    original_source_mac = source_mac_path.read_text()

    for v in args.bias_voltages_v:
        try:
            # select electric field from Meshes folder
            electric_field = get_electric_field(v)
            # replace bias voltage and distance in allpix simulation
            conf_text = set_conf_bias_and_output(
                original_conf,
                v,
                args.distance_mm,
                args.material,
                electric_field,
            )

            # write allpix config into conf_path using conf_text
            conf_path.write_text(conf_text)

            # copies the original source mac file into a variable
            source_mac_path.write_text(original_source_mac)
            # runs function to replace the distance in the original with the distance argument from command line
            set_macro_source(source_mac_path, args.distance_mm)

            # goes into allpix output and selects the correct ROOT file
            root_out = (
                Path("/home/claire/allpix-squared/output")
                / f"Am241alpha_{args.material}_{int(round(args.distance_mm))}mm_{int(round(v))}V.root"
            )

            print(f"ROOT_EDITOR>>>>>>root_out file is {root_out}")

            # RUN ALLPIX SIMULATION
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

        # restore the original source and config files
        finally:
            conf_path.write_text(original_conf)
            source_mac_path.write_text(original_source_mac)


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

    # sets silicon or silicon carbide detector
    parser.add_argument(
        "--material",
        default="Silicon_Carbide",
        help="Material used in Allpix Simulation",
    )

    args = parser.parse_args()

    # make variables for all the different scripts
    conf_path = Path(args.conf)
    source_mac_path = Path(args.source_mac)

    # pass the command line inputs bias voltage and distance, the config path, source path

    run_bias_scan(args, conf_path, source_mac_path)


if __name__ == "__main__":
    main()
