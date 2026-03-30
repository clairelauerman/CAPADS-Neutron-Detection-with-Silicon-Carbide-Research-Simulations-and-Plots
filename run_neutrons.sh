#!/bin/bash
#bash script that runs the neutron allpix squared simulation and 
#ROOT macro file
set -euo pipefail

ALLPIX_DIR="/home/claire/allpix-squared"
CONFIG="$1"
DISTANCE=$(printf "%.0f" "$2")

ROOT_OUT_DEFAULT="${ALLPIX_DIR}/output/Neutrons_{DISTANCE}mm_{ENERGY}MeV.root"
ROOT_OUT="${3:-$ROOT_OUT_DEFAULT}"
MACRO="/home/claire/allpix-squared/SiC_3x3/fit_collected_charge.C"

echo "CONFIG is ... ${CONFIG}"
echo "BASH_SCRIPT:> > > > > ROOT_OUT is ... ${ROOT_OUT}"

cd "${ALLPIX_DIR}"  
allpix -c "${CONFIG}"

#Run the ROOT macro file, which is fit_collected_charge
#root -l -b -q "${MACRO}(\"${ROOT_OUT}\", 7.83)"
