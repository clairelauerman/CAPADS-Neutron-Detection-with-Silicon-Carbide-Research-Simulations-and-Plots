#!/bin/bash
#bash script that runs the allpix squared simulation for alphas and
#the ROOT macro code
set -euo pipefail

ALLPIX_DIR="/home/claire/allpix-squared"
CONFIG="$1"
DISTANCE=$(printf "%.0f" "$2")

ROOT_OUT_DEFAULT="${ALLPIX_DIR}/output/Am241alpha_SiC_collimation_${DISTANCE}mm.root"
ROOT_OUT="${3:-$ROOT_OUT_DEFAULT}"
MACRO="/home/claire/allpix-squared/SiC_3x3/fit_collected_charge.C"

echo "CONFIG is ... ${CONFIG}"
echo "BASH_SCRIPT:> > > > > ROOT_OUT is ... ${ROOT_OUT}"

cd "${ALLPIX_DIR}"  
allpix -c "${CONFIG}"

#Run the ROOT macro file, which is fit_collected_charge
#root -l -b -q "${MACRO}(\"${ROOT_OUT}\", 3.9)"

