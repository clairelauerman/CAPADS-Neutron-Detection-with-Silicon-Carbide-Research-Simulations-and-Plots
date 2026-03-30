#!/usr/bin/env bash
#bash script that converts the electrostatic TCAD fields to electric fields
set -euo pipefail

ALLPIX_DIR="/home/claire/allpix-squared"
IN_DIR="${ALLPIX_DIR}/Neutrons/Meshes"
OUT_DIR="${ALLPIX_DIR}/Neutrons/Meshes"
P2F_BIN="${ALLPIX_DIR}/build/tools/apf_tools/potential_to_field"
UNITS="V/mm"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${P2F_BIN}" ]]; then
  echo "potential_to_field not found or not executable: ${P2F_BIN}"
  exit 1
fi

shopt -s nullglob
for f in "${IN_DIR}"/*_ElectrostaticPotential.apf; do
  base="$(basename "${f}" _ElectrostaticPotential.apf)"
  out="${OUT_DIR}/${base}_ElectricField.apf"
  echo "Converting ${f} -> ${out}"
  "${P2F_BIN}" --input "${f}" --output "${out}" --to apf --units "${UNITS}"
done
